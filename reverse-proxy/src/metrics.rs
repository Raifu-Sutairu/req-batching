use lazy_static::lazy_static;
use prometheus::{
    register_counter_vec, register_histogram_vec,
    register_gauge, CounterVec, HistogramVec, Gauge
};

lazy_static! {
    pub static ref FLUSH_COUNTER: CounterVec =
        register_counter_vec!(
            "batch_flush_total",
            "Total batch flushes by reason",
            &["reason"]  // "size_cap" | "timeout" | "rl_agent" | "rl_agent_timeout" | "rl_agent_error"
        ).unwrap();

    pub static ref BATCH_SIZE_HISTOGRAM: HistogramVec =
        register_histogram_vec!(
            "batch_size_at_flush",
            "Batch size when flushed",
            &["reason"],
            vec![1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 128.0]
        ).unwrap();

    pub static ref BATCH_AGE_HISTOGRAM: HistogramVec =
        register_histogram_vec!(
            "batch_age_ms_at_flush",
            "Batch age in ms when flushed",
            &["reason"],
            vec![5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        ).unwrap();

    pub static ref ACTIVE_BATCHES: Gauge =
        register_gauge!(
            "active_batch_slots",
            "Current number of open batch slots"
        ).unwrap();
}
