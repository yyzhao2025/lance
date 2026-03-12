use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::{self, Write};
use std::path::{Path as StdPath, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use futures::TryStreamExt;
use lance::Dataset;
use lance::io::ObjectStore;
use lance_arrow::iter_str_array;
use lance_core::cache::LanceCache;
use lance_index::Index;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::prefilter::NoFilter;
use lance_index::scalar::inverted::query::{
    FtsSearchParams, Operator, collect_query_tokens,
};
use lance_index::scalar::inverted::{InvertedIndex, InvertedIndexBuilder, InvertedIndexParams};
use lance_index::scalar::lance_format::LanceIndexStore;
use object_store::path::Path;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

const DEFAULT_TOKEN_COUNTS: &[usize] = &[1, 2, 3, 10, 20, 40, 80];
const DEFAULT_QUERIES_PER_COUNT: usize = 50;
const DEFAULT_QUERY_LIMIT: usize = 10;
const DEFAULT_MEASURE_REPEATS: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QueryGroup {
    token_count: usize,
    queries: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuerySet {
    dataset_uri: String,
    text_column: String,
    seed: u64,
    groups: Vec<QueryGroup>,
}

#[derive(Debug, Serialize)]
struct GroupBenchResult {
    token_count: usize,
    query_count: usize,
    avg_latency_ms: f64,
    p99_latency_ms: f64,
    min_latency_ms: f64,
    max_latency_ms: f64,
    avg_hits: f64,
}

#[derive(Debug, Serialize)]
struct BenchResult {
    label: String,
    dataset_uri: String,
    text_column: String,
    index_duration_secs: f64,
    query_limit: usize,
    warmup_queries: usize,
    repeats: usize,
    concurrency: usize,
    partition_count: usize,
    partition_total_bytes: u64,
    partition_min_bytes: u64,
    partition_max_bytes: u64,
    partition_avg_bytes: f64,
    partition_median_bytes: f64,
    groups: Vec<GroupBenchResult>,
}

#[derive(Debug)]
struct IndexStats {
    partition_count: usize,
    partition_total_bytes: u64,
    partition_min_bytes: u64,
    partition_max_bytes: u64,
    partition_avg_bytes: f64,
    partition_median_bytes: f64,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = env_logger::try_init();
    let mode = std::env::var("REAL_FTS_MODE").unwrap_or_else(|_| "sample".to_string());
    let dataset_uri = std::env::var("REAL_FTS_DATASET_URI")?;

    match mode.as_str() {
        "sample" => sample_queries(&dataset_uri).await?,
        "index_query" => benchmark_index_and_query(&dataset_uri).await?,
        "query_only" => benchmark_query_only(&dataset_uri).await?,
        other => {
            return Err(format!(
                "unsupported REAL_FTS_MODE={other}, expected sample, index_query, or query_only"
            )
            .into())
        }
    }

    Ok(())
}

async fn sample_queries(dataset_uri: &str) -> Result<(), Box<dyn std::error::Error>> {
    let text_column = std::env::var("REAL_FTS_TEXT_COLUMN")?;
    let query_file = std::env::var("REAL_FTS_QUERY_FILE")?;
    let queries_per_count = std::env::var("REAL_FTS_QUERIES_PER_COUNT")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(DEFAULT_QUERIES_PER_COUNT);
    let token_counts = parse_token_counts()?;
    let seed = std::env::var("REAL_FTS_QUERY_SEED")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(42u64);

    let dataset = Dataset::open(dataset_uri).await?;
    let mut scanner = dataset.scan();
    scanner.project(&[text_column.as_str()])?;
    scanner.batch_size(8_192);
    let mut stream = scanner.try_into_stream().await?;

    let params = InvertedIndexParams::default().with_position(false);
    let mut tokenizer = params.build()?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut groups: BTreeMap<usize, Vec<String>> = token_counts
        .iter()
        .copied()
        .map(|count| (count, Vec::with_capacity(queries_per_count)))
        .collect();
    let mut seen: HashMap<usize, BTreeSet<String>> = token_counts
        .iter()
        .copied()
        .map(|count| (count, BTreeSet::new()))
        .collect();

    while let Some(batch) = stream.try_next().await? {
        let doc_col = batch.column_by_name(&text_column).ok_or_else(|| {
            format!("column {text_column} not found in sampling batch")
        })?;

        for text in iter_str_array(doc_col.as_ref()) {
            let Some(text) = text else {
                continue;
            };
            let tokens = tokenize_doc(text, &mut tokenizer);
            if tokens.is_empty() {
                continue;
            }
            for &token_count in &token_counts {
                let Some(queries) = groups.get_mut(&token_count) else {
                    continue;
                };
                if queries.len() >= queries_per_count || tokens.len() < token_count {
                    continue;
                }
                if let Some(query) = sample_query(&tokens, token_count, &mut rng) {
                    let seen_for_count = seen
                        .get_mut(&token_count)
                        .expect("seen state should match group state");
                    if seen_for_count.insert(query.clone()) {
                        queries.push(query);
                    }
                }
            }
        }

        if groups.values().all(|queries| queries.len() >= queries_per_count) {
            break;
        }
    }

    let incomplete = groups
        .iter()
        .filter_map(|(token_count, queries)| {
            if queries.len() < queries_per_count {
                Some(format!(
                    "token_count={} have={} need={}",
                    token_count,
                    queries.len(),
                    queries_per_count
                ))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if !incomplete.is_empty() {
        return Err(format!("failed to sample enough queries: {}", incomplete.join(", ")).into());
    }

    let query_set = QuerySet {
        dataset_uri: dataset_uri.to_string(),
        text_column,
        seed,
        groups: token_counts
            .into_iter()
            .map(|token_count| QueryGroup {
                token_count,
                queries: groups.remove(&token_count).unwrap_or_default(),
            })
            .collect(),
    };

    if let Some(parent) = StdPath::new(&query_file).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&query_file, serde_json::to_vec_pretty(&query_set)?)?;
    println!("QUERY_FILE {}", query_file);
    println!("RESULT sampled_groups={}", query_set.groups.len());
    println!(
        "RESULT sampled_queries={}",
        query_set.groups.iter().map(|group| group.queries.len()).sum::<usize>()
    );
    io::stdout().flush()?;
    Ok(())
}

async fn benchmark_index_and_query(dataset_uri: &str) -> Result<(), Box<dyn std::error::Error>> {
    let text_column = std::env::var("REAL_FTS_TEXT_COLUMN")?;
    let query_file = std::env::var("REAL_FTS_QUERY_FILE")?;
    let label = std::env::var("REAL_FTS_LABEL").unwrap_or_else(|_| "bench".to_string());
    let batch_size = std::env::var("REAL_FTS_BATCH_SIZE")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(8_192);
    let query_limit = std::env::var("REAL_FTS_QUERY_LIMIT")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(DEFAULT_QUERY_LIMIT);
    let repeats = std::env::var("REAL_FTS_QUERY_REPEATS")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(DEFAULT_MEASURE_REPEATS);
    let concurrency = std::env::var("REAL_FTS_QUERY_CONCURRENCY")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(1usize)
        .max(1);
    let index_root = index_root()?;

    let query_set: QuerySet = serde_json::from_slice(&fs::read(&query_file)?)?;

    let index_dir = Path::from_filesystem_path(&index_root)?;
    let store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        index_dir,
        Arc::new(LanceCache::no_cache()),
    ));

    let dataset = Dataset::open(dataset_uri).await?;
    let mut scanner = dataset.scan();
    scanner.project(&[text_column.as_str()])?;
    scanner.batch_size(batch_size);
    scanner.with_row_id();
    let stream = scanner.try_into_stream().await?;

    let mut builder =
        InvertedIndexBuilder::new(InvertedIndexParams::default().with_position(false));

    println!("READY");
    io::stdout().flush()?;

    let build_start = Instant::now();
    builder.update(stream.into(), store.as_ref()).await?;
    let index_duration = build_start.elapsed();

    let stats = collect_index_stats(index_root.as_path())?;

    let cache = Arc::new(LanceCache::with_capacity(4096));
    let index = InvertedIndex::load(store, None, cache.as_ref()).await?;
    let result = benchmark_loaded_index(
        index,
        stats,
        query_set,
        label,
        dataset_uri.to_string(),
        query_limit,
        repeats,
        concurrency,
        index_duration.as_secs_f64(),
    )
    .await?;

    println!("RESULT_JSON {}", serde_json::to_string(&result)?);
    println!("INDEX_DIR {}", index_root.display());
    io::stdout().flush()?;
    Ok(())
}

async fn benchmark_query_only(dataset_uri: &str) -> Result<(), Box<dyn std::error::Error>> {
    let query_file = std::env::var("REAL_FTS_QUERY_FILE")?;
    let label = std::env::var("REAL_FTS_LABEL").unwrap_or_else(|_| "bench".to_string());
    let query_limit = std::env::var("REAL_FTS_QUERY_LIMIT")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(DEFAULT_QUERY_LIMIT);
    let repeats = std::env::var("REAL_FTS_QUERY_REPEATS")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(DEFAULT_MEASURE_REPEATS);
    let concurrency = std::env::var("REAL_FTS_QUERY_CONCURRENCY")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(1usize)
        .max(1);
    let index_root = PathBuf::from(std::env::var("REAL_FTS_INDEX_DIR")?);
    let query_set: QuerySet = serde_json::from_slice(&fs::read(&query_file)?)?;
    let stats = collect_index_stats(index_root.as_path())?;
    let index = load_index(index_root.as_path()).await?;

    let result = benchmark_loaded_index(
        index,
        stats,
        query_set,
        label,
        dataset_uri.to_string(),
        query_limit,
        repeats,
        concurrency,
        0.0,
    )
    .await?;

    println!("RESULT_JSON {}", serde_json::to_string(&result)?);
    println!("INDEX_DIR {}", index_root.display());
    io::stdout().flush()?;
    Ok(())
}

async fn benchmark_loaded_index(
    index: Arc<InvertedIndex>,
    stats: IndexStats,
    query_set: QuerySet,
    label: String,
    dataset_uri: String,
    query_limit: usize,
    repeats: usize,
    concurrency: usize,
    index_duration_secs: f64,
) -> Result<BenchResult, Box<dyn std::error::Error>> {
    index.prewarm().await?;

    let params = Arc::new(FtsSearchParams::new().with_limit(Some(query_limit)));
    let prefilter = Arc::new(NoFilter);
    let metrics = Arc::new(NoOpMetricsCollector);
    let mut tokenizer = index.tokenizer();

    let mut group_results = Vec::with_capacity(query_set.groups.len());
    let mut warmup_queries = 0usize;

    for group in &query_set.groups {
        let prepared_queries = group
            .queries
            .iter()
            .map(|query| Arc::new(collect_query_tokens(query, &mut tokenizer)))
            .collect::<Vec<_>>();

        for tokens in &prepared_queries {
            let _ = index
                .bm25_search(
                    tokens.clone(),
                    params.clone(),
                    Operator::Or,
                    prefilter.clone(),
                    metrics.clone(),
                )
                .await?;
            warmup_queries += 1;
        }

        let mut latencies_ms = Vec::with_capacity(prepared_queries.len() * repeats);
        let mut total_hits = 0usize;
        for _ in 0..repeats {
            for chunk in prepared_queries.chunks(concurrency) {
                let mut handles = Vec::with_capacity(chunk.len());
                for tokens in chunk {
                    let index = index.clone();
                    let params = params.clone();
                    let prefilter = prefilter.clone();
                    let metrics = metrics.clone();
                    let tokens = tokens.clone();
                    let submitted = Instant::now();
                    handles.push(tokio::spawn(async move {
                        let (row_ids, _scores) = index
                            .bm25_search(tokens, params, Operator::Or, prefilter, metrics)
                            .await?;
                        Ok::<(f64, usize), lance_core::Error>((
                            submitted.elapsed().as_secs_f64() * 1000.0,
                            row_ids.len(),
                        ))
                    }));
                }

                for handle in handles {
                    let (latency_ms, hits) = handle.await??;
                    latencies_ms.push(latency_ms);
                    total_hits += hits;
                }
            }
        }

        let latency_sum = latencies_ms.iter().copied().sum::<f64>();
        let avg_latency_ms = latency_sum / latencies_ms.len() as f64;
        let p99_latency_ms = percentile(&latencies_ms, 0.99);
        let min_latency_ms = latencies_ms
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max_latency_ms = latencies_ms.iter().copied().fold(0.0, f64::max);
        let avg_hits = total_hits as f64 / latencies_ms.len() as f64;

        group_results.push(GroupBenchResult {
            token_count: group.token_count,
            query_count: prepared_queries.len(),
            avg_latency_ms,
            p99_latency_ms,
            min_latency_ms,
            max_latency_ms,
            avg_hits,
        });
    }

    Ok(BenchResult {
        label,
        dataset_uri,
        text_column: query_set.text_column,
        index_duration_secs,
        query_limit,
        warmup_queries,
        repeats,
        concurrency,
        partition_count: stats.partition_count,
        partition_total_bytes: stats.partition_total_bytes,
        partition_min_bytes: stats.partition_min_bytes,
        partition_max_bytes: stats.partition_max_bytes,
        partition_avg_bytes: stats.partition_avg_bytes,
        partition_median_bytes: stats.partition_median_bytes,
        groups: group_results,
    })
}

fn sample_query(tokens: &[String], token_count: usize, rng: &mut StdRng) -> Option<String> {
    if tokens.len() < token_count {
        return None;
    }
    let max_start = tokens.len() - token_count;
    let start = if max_start == 0 {
        0
    } else {
        rng.random_range(0..=max_start)
    };
    Some(tokens[start..start + token_count].join(" "))
}

fn tokenize_doc(
    text: &str,
    tokenizer: &mut Box<dyn lance_index::scalar::inverted::lance_tokenizer::LanceTokenizer>,
) -> Vec<String> {
    let mut stream = tokenizer.token_stream_for_doc(text);
    let mut tokens = Vec::new();
    while let Some(token) = stream.next() {
        tokens.push(token.text.clone());
    }
    tokens
}

fn collect_index_stats(index_root: &StdPath) -> Result<IndexStats, Box<dyn std::error::Error>> {
    let mut partition_sizes = BTreeMap::<u64, u64>::new();

    for entry in fs::read_dir(index_root)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_file() {
            continue;
        }
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        let Some(partition_id) = parse_partition_id(&file_name) else {
            continue;
        };
        let file_size = entry.metadata()?.len();
        *partition_sizes.entry(partition_id).or_default() += file_size;
    }

    let mut sizes = partition_sizes.into_values().collect::<Vec<_>>();
    sizes.sort_unstable();

    let partition_count = sizes.len();
    let partition_total_bytes = sizes.iter().copied().sum::<u64>();
    let partition_min_bytes = sizes.first().copied().unwrap_or_default();
    let partition_max_bytes = sizes.last().copied().unwrap_or_default();
    let partition_avg_bytes = if partition_count == 0 {
        0.0
    } else {
        partition_total_bytes as f64 / partition_count as f64
    };
    let partition_median_bytes = if partition_count == 0 {
        0.0
    } else if partition_count % 2 == 1 {
        sizes[partition_count / 2] as f64
    } else {
        let upper = sizes[partition_count / 2] as f64;
        let lower = sizes[(partition_count / 2) - 1] as f64;
        (lower + upper) / 2.0
    };

    Ok(IndexStats {
        partition_count,
        partition_total_bytes,
        partition_min_bytes,
        partition_max_bytes,
        partition_avg_bytes,
        partition_median_bytes,
    })
}

async fn load_index(index_root: &StdPath) -> Result<Arc<InvertedIndex>, Box<dyn std::error::Error>> {
    let index_dir = Path::from_filesystem_path(index_root)?;
    let store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        index_dir,
        Arc::new(LanceCache::no_cache()),
    ));
    let cache = Arc::new(LanceCache::with_capacity(4096));
    Ok(InvertedIndex::load(store, None, cache.as_ref()).await?)
}

fn parse_partition_id(file_name: &str) -> Option<u64> {
    let suffix = file_name.strip_prefix("part_")?;
    let (partition_id, _) = suffix.split_once('_')?;
    partition_id.parse().ok()
}

fn percentile(values: &[f64], q: f64) -> f64 {
    assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let idx = ((sorted.len() - 1) as f64 * q).ceil() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn parse_token_counts() -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let token_counts = std::env::var("REAL_FTS_QUERY_TOKEN_COUNTS")
        .unwrap_or_else(|_| {
            DEFAULT_TOKEN_COUNTS
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;

    if token_counts.is_empty() {
        return Err("REAL_FTS_QUERY_TOKEN_COUNTS must not be empty".into());
    }
    Ok(token_counts)
}

fn index_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = if let Ok(dir) = std::env::var("REAL_FTS_INDEX_DIR") {
        let dir = PathBuf::from(dir);
        if dir.exists() {
            fs::remove_dir_all(&dir)?;
        }
        fs::create_dir_all(&dir)?;
        dir
    } else {
        tempfile::tempdir()?.keep()
    };
    Ok(path)
}
