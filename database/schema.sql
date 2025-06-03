-- ============================================================================
-- Power of 3 Trading System - PostgreSQL Database Schema
-- ============================================================================
-- 
-- This schema tracks:
-- - Trading signals and executions
-- - Performance metrics and analytics
-- - Risk management data
-- - Market analysis results
-- - Session-based statistics
--
-- Created for Power of 3 ICT methodology tracking
-- ============================================================================

-- Drop existing tables if they exist (for clean setup)
DROP TABLE IF EXISTS trade_exits CASCADE;
DROP TABLE IF EXISTS trade_entries CASCADE;
DROP TABLE IF EXISTS signals CASCADE;
DROP TABLE IF EXISTS manipulation_patterns CASCADE;
DROP TABLE IF EXISTS liquidity_zones CASCADE;
DROP TABLE IF EXISTS market_analysis CASCADE;
DROP TABLE IF EXISTS trading_sessions CASCADE;
DROP TABLE IF EXISTS performance_metrics CASCADE;
DROP TABLE IF EXISTS risk_management CASCADE;
DROP TABLE IF EXISTS account_balance CASCADE;

-- Enable UUID extension for unique IDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- 1. ACCOUNT BALANCE TABLE
-- ============================================================================
CREATE TABLE account_balance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    balance_date DATE NOT NULL,
    starting_balance DECIMAL(12,2) NOT NULL,
    ending_balance DECIMAL(12,2) NOT NULL,
    daily_pnl DECIMAL(12,2) NOT NULL DEFAULT 0.00,
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- 2. TRADING SESSIONS TABLE
-- ============================================================================
CREATE TABLE trading_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_date DATE NOT NULL,
    session_type VARCHAR(20) NOT NULL CHECK (session_type IN ('london_open', 'new_york_open', 'london_close')),
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    phase VARCHAR(20) CHECK (phase IN ('accumulation', 'manipulation', 'direction')),
    total_signals INTEGER DEFAULT 0,
    executed_trades INTEGER DEFAULT 0,
    session_pnl DECIMAL(10,2) DEFAULT 0.00,
    volatility_score DECIMAL(3,2), -- 0.00 to 1.00
    market_conditions TEXT,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- 3. MARKET ANALYSIS TABLE
-- ============================================================================
CREATE TABLE market_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    analysis_time TIMESTAMP WITH TIME ZONE NOT NULL,
    session_id UUID REFERENCES trading_sessions(id),
    market_structure VARCHAR(20) NOT NULL CHECK (market_structure IN ('bullish', 'bearish', 'neutral')),
    trend_strength DECIMAL(3,2), -- 0.00 to 1.00
    higher_highs BOOLEAN,
    higher_lows BOOLEAN,
    current_price DECIMAL(10,4) NOT NULL,
    support_level DECIMAL(10,4),
    resistance_level DECIMAL(10,4),
    volume_analysis TEXT,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- 4. LIQUIDITY ZONES TABLE
-- ============================================================================
CREATE TABLE liquidity_zones (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    session_id UUID REFERENCES trading_sessions(id),
    market_analysis_id UUID REFERENCES market_analysis(id),
    zone_level DECIMAL(10,4) NOT NULL,
    zone_type VARCHAR(20) NOT NULL CHECK (zone_type IN ('support', 'resistance')),
    strength DECIMAL(3,2) NOT NULL, -- 0.00 to 1.00
    touches INTEGER DEFAULT 0,
    last_test TIMESTAMP WITH TIME ZONE,
    is_broken BOOLEAN DEFAULT FALSE,
    break_time TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- 5. MANIPULATION PATTERNS TABLE
-- ============================================================================
CREATE TABLE manipulation_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    session_id UUID REFERENCES trading_sessions(id),
    pattern_type VARCHAR(30) NOT NULL CHECK (pattern_type IN ('liquidity_sweep', 'fake_breakout', 'stop_hunt')),
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    trigger_level DECIMAL(10,4) NOT NULL,
    reversal_level DECIMAL(10,4) NOT NULL,
    strength DECIMAL(3,2) NOT NULL, -- 0.00 to 1.00
    volume_surge BOOLEAN DEFAULT FALSE,
    price_move_pips DECIMAL(8,2),
    reversal_pips DECIMAL(8,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- 6. SIGNALS TABLE
-- ============================================================================
CREATE TABLE signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id VARCHAR(100) UNIQUE NOT NULL, -- From Power of 3 system
    symbol VARCHAR(10) NOT NULL,
    signal_type VARCHAR(10) NOT NULL CHECK (signal_type IN ('long', 'short')),
    session_id UUID REFERENCES trading_sessions(id),
    manipulation_pattern_id UUID REFERENCES manipulation_patterns(id),
    
    -- Signal Quality
    quality_score DECIMAL(4,2) NOT NULL, -- 0.00 to 10.00
    quality_level VARCHAR(20) NOT NULL CHECK (quality_level IN ('excellent', 'good', 'average', 'poor')),
    confidence DECIMAL(3,2) NOT NULL, -- 0.00 to 1.00
    
    -- Entry Details
    entry_price DECIMAL(10,4) NOT NULL,
    stop_loss DECIMAL(10,4) NOT NULL,
    take_profit_1 DECIMAL(10,4) NOT NULL,
    take_profit_2 DECIMAL(10,4) NOT NULL,
    take_profit_3 DECIMAL(10,4) NOT NULL,
    
    -- Risk Management
    risk_reward_ratio DECIMAL(5,2) NOT NULL,
    position_size DECIMAL(12,6) NOT NULL,
    risk_amount DECIMAL(10,2) NOT NULL,
    
    -- Timing
    generated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Status
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'executed', 'expired', 'cancelled')),
    execution_reason TEXT,
    notes TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- 7. TRADE ENTRIES TABLE
-- ============================================================================
CREATE TABLE trade_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID REFERENCES signals(id),
    symbol VARCHAR(10) NOT NULL,
    trade_type VARCHAR(10) NOT NULL CHECK (trade_type IN ('long', 'short')),
    
    -- Entry Details
    entry_price DECIMAL(10,4) NOT NULL,
    entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
    position_size DECIMAL(12,6) NOT NULL,
    stop_loss DECIMAL(10,4) NOT NULL,
    take_profit_1 DECIMAL(10,4) NOT NULL,
    take_profit_2 DECIMAL(10,4),
    take_profit_3 DECIMAL(10,4),
    
    -- Risk Management
    risk_amount DECIMAL(10,2) NOT NULL,
    account_balance_at_entry DECIMAL(12,2) NOT NULL,
    risk_percentage DECIMAL(5,2) NOT NULL,
    
    -- Execution Details
    execution_method VARCHAR(20) DEFAULT 'manual' CHECK (execution_method IN ('manual', 'automated')),
    slippage_pips DECIMAL(6,2) DEFAULT 0.00,
    commission DECIMAL(8,2) DEFAULT 0.00,
    spread DECIMAL(6,2),
    
    -- Status
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'partially_closed', 'closed', 'stopped_out')),
    
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- 8. TRADE EXITS TABLE
-- ============================================================================
CREATE TABLE trade_exits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trade_entry_id UUID REFERENCES trade_entries(id),
    exit_type VARCHAR(20) NOT NULL CHECK (exit_type IN ('take_profit_1', 'take_profit_2', 'take_profit_3', 'stop_loss', 'manual_close', 'break_even')),
    
    -- Exit Details
    exit_price DECIMAL(10,4) NOT NULL,
    exit_time TIMESTAMP WITH TIME ZONE NOT NULL,
    position_size_closed DECIMAL(12,6) NOT NULL,
    remaining_position DECIMAL(12,6) DEFAULT 0.00,
    
    -- P&L Calculation
    gross_pnl DECIMAL(10,2) NOT NULL,
    commission DECIMAL(8,2) DEFAULT 0.00,
    net_pnl DECIMAL(10,2) NOT NULL,
    pips_gained DECIMAL(8,2),
    percentage_return DECIMAL(6,2),
    
    -- Execution Details
    slippage_pips DECIMAL(6,2) DEFAULT 0.00,
    execution_method VARCHAR(20) DEFAULT 'manual',
    
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- 9. RISK MANAGEMENT TABLE
-- ============================================================================
CREATE TABLE risk_management (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL,
    account_balance DECIMAL(12,2) NOT NULL,
    total_risk_amount DECIMAL(10,2) NOT NULL,
    max_daily_risk DECIMAL(10,2) NOT NULL,
    current_daily_risk DECIMAL(10,2) DEFAULT 0.00,
    open_positions INTEGER DEFAULT 0,
    max_positions INTEGER DEFAULT 3,
    correlation_limit DECIMAL(3,2) DEFAULT 0.70,
    drawdown_limit DECIMAL(3,2) DEFAULT 0.05, -- 5%
    current_drawdown DECIMAL(3,2) DEFAULT 0.00,
    risk_status VARCHAR(20) DEFAULT 'normal' CHECK (risk_status IN ('normal', 'elevated', 'maximum', 'halt')),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- 10. PERFORMANCE METRICS TABLE
-- ============================================================================
CREATE TABLE performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    period_type VARCHAR(20) NOT NULL CHECK (period_type IN ('daily', 'weekly', 'monthly', 'yearly')),
    
    -- Account Metrics
    starting_balance DECIMAL(12,2) NOT NULL,
    ending_balance DECIMAL(12,2) NOT NULL,
    net_profit DECIMAL(12,2) NOT NULL,
    gross_profit DECIMAL(12,2) NOT NULL,
    gross_loss DECIMAL(12,2) NOT NULL,
    
    -- Trading Metrics
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    win_rate DECIMAL(5,2) NOT NULL DEFAULT 0.00, -- Percentage
    
    -- Risk Metrics
    average_win DECIMAL(10,2) DEFAULT 0.00,
    average_loss DECIMAL(10,2) DEFAULT 0.00,
    largest_win DECIMAL(10,2) DEFAULT 0.00,
    largest_loss DECIMAL(10,2) DEFAULT 0.00,
    profit_factor DECIMAL(6,2) DEFAULT 0.00, -- Gross Profit / Gross Loss
    
    -- Advanced Metrics
    sharpe_ratio DECIMAL(6,3),
    max_drawdown DECIMAL(10,2) DEFAULT 0.00,
    max_drawdown_percentage DECIMAL(5,2) DEFAULT 0.00,
    recovery_factor DECIMAL(6,2),
    
    -- Session Breakdown
    london_open_pnl DECIMAL(10,2) DEFAULT 0.00,
    new_york_open_pnl DECIMAL(10,2) DEFAULT 0.00,
    london_close_pnl DECIMAL(10,2) DEFAULT 0.00,
    
    -- Power of 3 Specific
    manipulation_signals INTEGER DEFAULT 0,
    continuation_signals INTEGER DEFAULT 0,
    avg_signal_quality DECIMAL(4,2) DEFAULT 0.00,
    avg_risk_reward DECIMAL(5,2) DEFAULT 0.00,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Frequently queried columns
CREATE INDEX idx_signals_symbol_date ON signals(symbol, generated_at);
CREATE INDEX idx_signals_session ON signals(session_id);
CREATE INDEX idx_signals_status ON signals(status);
CREATE INDEX idx_signals_quality ON signals(quality_score DESC);

CREATE INDEX idx_trade_entries_symbol_date ON trade_entries(symbol, entry_time);
CREATE INDEX idx_trade_entries_status ON trade_entries(status);
CREATE INDEX idx_trade_entries_signal ON trade_entries(signal_id);

CREATE INDEX idx_trade_exits_entry_id ON trade_exits(trade_entry_id);
CREATE INDEX idx_trade_exits_date ON trade_exits(exit_time);
CREATE INDEX idx_trade_exits_type ON trade_exits(exit_type);

CREATE INDEX idx_sessions_date_type ON trading_sessions(session_date, session_type);
CREATE INDEX idx_market_analysis_symbol_time ON market_analysis(symbol, analysis_time);
CREATE INDEX idx_liquidity_zones_symbol_level ON liquidity_zones(symbol, zone_level);
CREATE INDEX idx_manipulation_patterns_symbol_time ON manipulation_patterns(symbol, start_time);

CREATE INDEX idx_performance_metrics_period ON performance_metrics(period_type, period_start, period_end);
CREATE INDEX idx_risk_management_date ON risk_management(date);
CREATE INDEX idx_account_balance_date ON account_balance(balance_date);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- ============================================================================

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to tables with updated_at columns
CREATE TRIGGER update_account_balance_updated_at BEFORE UPDATE ON account_balance 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_performance_metrics_updated_at BEFORE UPDATE ON performance_metrics 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: Open Trades Summary
CREATE VIEW open_trades_summary AS
SELECT 
    te.id,
    te.symbol,
    te.trade_type,
    te.entry_price,
    te.entry_time,
    te.position_size,
    te.stop_loss,
    te.risk_amount,
    s.quality_score,
    s.signal_type,
    (te.risk_amount / te.account_balance_at_entry * 100) as risk_percentage,
    CASE 
        WHEN te.trade_type = 'long' THEN (te.stop_loss - te.entry_price)
        ELSE (te.entry_price - te.stop_loss)
    END as max_loss_pips
FROM trade_entries te
LEFT JOIN signals s ON te.signal_id = s.id
WHERE te.status = 'open'
ORDER BY te.entry_time DESC;

-- View: Daily Performance Summary
CREATE VIEW daily_performance AS
SELECT 
    te.entry_time::DATE as trade_date,
    COUNT(tex.id) as closed_trades,
    SUM(CASE WHEN tex.net_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN tex.net_pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
    ROUND(AVG(CASE WHEN tex.net_pnl > 0 THEN tex.net_pnl END), 2) as avg_win,
    ROUND(AVG(CASE WHEN tex.net_pnl < 0 THEN tex.net_pnl END), 2) as avg_loss,
    ROUND(SUM(tex.net_pnl), 2) as daily_pnl,
    ROUND(AVG(s.quality_score), 2) as avg_signal_quality
FROM trade_entries te
JOIN trade_exits tex ON te.id = tex.trade_entry_id
LEFT JOIN signals s ON te.signal_id = s.id
WHERE tex.exit_type IN ('take_profit_1', 'take_profit_2', 'take_profit_3', 'stop_loss', 'manual_close')
GROUP BY te.entry_time::DATE
ORDER BY trade_date DESC;

-- View: Session Performance
CREATE VIEW session_performance AS
SELECT 
    ts.session_type,
    ts.session_date,
    COUNT(s.id) as signals_generated,
    COUNT(te.id) as trades_executed,
    SUM(COALESCE(tex.net_pnl, 0)) as session_pnl,
    AVG(s.quality_score) as avg_signal_quality,
    COUNT(mp.id) as manipulation_patterns
FROM trading_sessions ts
LEFT JOIN signals s ON ts.id = s.session_id
LEFT JOIN trade_entries te ON s.id = te.signal_id
LEFT JOIN trade_exits tex ON te.id = tex.trade_entry_id
LEFT JOIN manipulation_patterns mp ON ts.id = mp.session_id
GROUP BY ts.session_type, ts.session_date
ORDER BY ts.session_date DESC, ts.session_type;

-- View: Symbol Performance
CREATE VIEW symbol_performance AS
SELECT 
    te.symbol,
    COUNT(tex.id) as total_trades,
    SUM(CASE WHEN tex.net_pnl > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN tex.net_pnl < 0 THEN 1 ELSE 0 END) as losses,
    ROUND(AVG(CASE WHEN tex.net_pnl > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
    ROUND(SUM(tex.net_pnl), 2) as total_pnl,
    ROUND(AVG(tex.net_pnl), 2) as avg_pnl_per_trade,
    ROUND(AVG(s.quality_score), 2) as avg_signal_quality,
    MAX(tex.net_pnl) as best_trade,
    MIN(tex.net_pnl) as worst_trade
FROM trade_entries te
JOIN trade_exits tex ON te.id = tex.trade_entry_id
LEFT JOIN signals s ON te.signal_id = s.id
GROUP BY te.symbol
ORDER BY total_pnl DESC;

-- ============================================================================
-- SAMPLE DATA INSERTION FUNCTIONS
-- ============================================================================

-- Function to calculate win rate
CREATE OR REPLACE FUNCTION calculate_win_rate(wins INTEGER, total INTEGER)
RETURNS DECIMAL(5,2) AS $$
BEGIN
    IF total = 0 THEN
        RETURN 0.00;
    END IF;
    RETURN ROUND((wins::DECIMAL / total::DECIMAL) * 100, 2);
END;
$$ LANGUAGE plpgsql;

-- Function to calculate profit factor
CREATE OR REPLACE FUNCTION calculate_profit_factor(gross_profit DECIMAL, gross_loss DECIMAL)
RETURNS DECIMAL(6,2) AS $$
BEGIN
    IF gross_loss = 0 THEN
        RETURN 999.99; -- Avoid division by zero
    END IF;
    RETURN ROUND(gross_profit / ABS(gross_loss), 2);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SAMPLE QUERIES FOR REPORTING
-- ============================================================================

/*
-- Get trading performance for last 30 days
SELECT * FROM daily_performance 
WHERE trade_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY trade_date DESC;

-- Get best performing sessions
SELECT 
    session_type,
    COUNT(*) as sessions,
    AVG(session_pnl) as avg_pnl,
    SUM(session_pnl) as total_pnl
FROM session_performance 
WHERE session_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY session_type
ORDER BY avg_pnl DESC;

-- Get current open positions
SELECT * FROM open_trades_summary;

-- Get monthly performance summary
SELECT 
    DATE_TRUNC('month', period_start) as month,
    net_profit,
    win_rate,
    total_trades,
    avg_signal_quality,
    max_drawdown_percentage
FROM performance_metrics 
WHERE period_type = 'monthly'
ORDER BY month DESC;

-- Get manipulation pattern effectiveness
SELECT 
    mp.pattern_type,
    COUNT(te.id) as trades_from_pattern,
    AVG(tex.net_pnl) as avg_pnl,
    SUM(tex.net_pnl) as total_pnl,
    AVG(s.quality_score) as avg_signal_quality
FROM manipulation_patterns mp
JOIN signals s ON mp.id = s.manipulation_pattern_id
JOIN trade_entries te ON s.id = te.signal_id
JOIN trade_exits tex ON te.id = tex.trade_entry_id
GROUP BY mp.pattern_type
ORDER BY avg_pnl DESC;
*/

-- ============================================================================
-- SCHEMA CREATION COMPLETE
-- ============================================================================

-- Display schema summary
SELECT 'Power of 3 Trading Database Schema Created Successfully!' as status,
       'Tables: ' || COUNT(*) as table_count
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_type = 'BASE TABLE';