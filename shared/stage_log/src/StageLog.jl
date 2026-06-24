module StageLog

import Logging
using Logging:
    AbstractLogger,
    BelowMinLevel,
    Warn,
    Error,
    handle_message,
    min_enabled_level,
    shouldlog,
    global_logger

export StageLogger, setup_logger!

"""
    StageLogger <: AbstractLogger

Logger that prefixes messages with a stage name (e.g., "[input]", "[assess]")
and writes to both stdout/stderr and a named log file.

Usage:
    using StageLog
    StageLog.setup_logger!("input", "input.log")
    @info "hello"  # → "[input] hello" to stdout + input.log
"""
struct StageLogger <: AbstractLogger
    prefix::String
    io::IOStream
end

function StageLogger(prefix::String, filename::AbstractString)
    return StageLogger(prefix, open(filename, "w"))
end

Logging.min_enabled_level(::StageLogger) = BelowMinLevel
Logging.shouldlog(::StageLogger, args...) = true

function Logging.handle_message(
    logger::StageLogger,
    lvl,
    msg,
    _mod,
    group,
    id,
    file,
    line;
    kwargs...,
)
    pfx = lvl == Warn ? "WARN: " : lvl == Error ? "ERROR: " : ""
    line_str = "[$(logger.prefix)] $pfx$msg"

    if lvl >= Warn
        println(stderr, line_str)
    else
        println(stdout, line_str)
    end
    println(logger.io, line_str)
    flush(logger.io)
end

"""
    setup_logger!(prefix, filename)

Create a StageLogger with the given `prefix` and log `filename`, set it as the
global logger, and register an `atexit` handler to close the log file on exit.
"""
function setup_logger!(prefix::String, filename::AbstractString)
    logger = StageLogger(prefix, filename)
    global_logger(logger)
    atexit() do
        close(logger.io)
    end
end

end # module
