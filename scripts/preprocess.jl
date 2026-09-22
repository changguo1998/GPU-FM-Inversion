#!/usr/bin/env julia
#
# preprocess.jl - 试次生成 (trial generation)
#
# 读最新 status_N.h5 的 /strategy（完整网格定义 + depth/freq/duration 索引），
# 按预算生成笛卡尔积试次（离散参数以整数索引表示，物理值仅存于
# database.h5 /paraspace/depth，output 时按需解析），
# 写回同一 status_N.h5 的 /trials。forward 阶段随后消费 /trials。
#
# Usage:
#   DATA_DIR=<dir> julia scripts/preprocess.jl
#
# 无 CLI 参数；目录布局由 driver.sh 控制（DATA_DIR 导出，
# status_N.h5 位于 $DATA_DIR/status/）。

using HDF5

using StageLog

using IO, Search

# === 1. 定位文件 ===
data_dir = ENV["DATA_DIR"]
StageLog.setup_logger!("preprocess", joinpath(data_dir, "preprocess.log"))

status_dir = joinpath(data_dir, "status")

status_path, iter_n = IO.find_latest_status(status_dir)

@info "=" ^ 70
@info "preprocess stage started"
@info "  data dir     = $data_dir"
@info "  status file  = $status_path (iteration $iter_n)"

# === 2. 读策略（完整网格） ===
strategy = IO.read_strategy(status_path)

# === 3. 预算规划并生成试次 === (离散参数以整数索引表示; 物理值仅存于 /paraspace)
@info "  grid: strike $(strategy.nstrike) × dip $(strategy.ndip) × rake $(strategy.nrake) @ $(strategy.dstrike)°, depth $(length(strategy.depth_indices)), freq $(length(strategy.freq_indices)), duration $(length(strategy.duration_indices))"
full_trial_count = prod((
    max(Int(strategy.nstrike), 1),
    max(Int(strategy.ndip), 1),
    max(Int(strategy.nrake), 1),
    max(length(strategy.depth_indices), 1),
    max(length(strategy.freq_indices), 1),
    max(length(strategy.duration_indices), 1),
))
budget_text = get(ENV, "FM_TRIAL_BUDGET", "")
budget = if isempty(strip(budget_text))
    full_trial_count
else
    try
        parse(Int, strip(budget_text))
    catch
        error("FM_TRIAL_BUDGET must be a positive integer, got '$budget_text'")
    end
end
budget > 0 || error("FM_TRIAL_BUDGET must be a positive integer, got '$budget_text'")
plan = Search.budgeted_plan(strategy, budget)
budget_label = budget == full_trial_count ? "full ($full_trial_count)" : string(budget)
@info "  trial budget: $budget_label"
@info "  selected axes: strike $(length(plan.strike_indices)), dip $(length(plan.dip_indices)), rake $(length(plan.rake_indices)), depth $(length(plan.depth_indices)), freq $(length(plan.freq_indices)), duration $(length(plan.duration_indices))"
t0 = time()
trials = Search.generate_trials(plan)
elapsed = time() - t0

@info "  generated $(length(trials.strike_idx)) trials in $(round(elapsed, digits = 3)) s"

# === 5. 写 /trials ===
IO.write_trials(status_path, trials)
@info "  wrote /trials to $status_path"
