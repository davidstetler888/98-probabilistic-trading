# Win Rate Enhancement Plan
*Strategic approach to improve win rate from 45-55% to 60-70% while maintaining 35+ trades/week*

## Executive Summary

Your system has successfully achieved excellent trade frequency (~50 trades/period) and outstanding RR (3.20). The next phase focuses on **signal quality enhancement** to improve win rate consistency while maintaining your current trade volume and RR performance.

## Current Performance Assessment

### Strengths
- ✅ Trade frequency: 42-55 trades/period (target achieved)
- ✅ Risk-reward: Consistent 3.20 RR (excellent)
- ✅ Signal generation: 351+ signals/week
- ✅ Market regime detection: 3 regimes operational
- ✅ Session filtering: Implemented and active

### Target Improvements
- 🎯 Win rate: 45-55% → 60-70%
- 🎯 Win rate stability: Reduce volatility (current: 20-100% swings)
- 🎯 Maintain trade frequency: 35+ trades/week
- 🎯 Preserve RR: Keep above 2.5

## Phase 1: Signal Quality Enhancement (Week 1-2)

### 1.1 Enhanced Multi-Factor Signal Scoring

**Current Issue**: Basic edge scoring may miss signal quality nuances
**Solution**: Implement comprehensive signal quality assessment

```python
def calculate_enhanced_signal_quality(signal_data):
    """
    Calculate comprehensive signal quality score (0-100)
    """
    quality_score = 0
    
    # Base probability strength (0-30 points)
    base_prob = signal_data['meta_prob']
    quality_score += min(base_prob * 30, 30)
    
    # Market regime alignment (0-20 points)
    regime = signal_data['market_regime']
    regime_scores = {0: 20, 1: 15, 2: 10}  # Regime 0 typically best for signals
    quality_score += regime_scores.get(regime, 5)
    
    # Session quality bonus (0-15 points)
    hour = signal_data.index.hour
    if 8 <= hour < 13:      # London session
        quality_score += 15
    elif 13 <= hour < 18:   # NY session
        quality_score += 12
    elif 12 <= hour < 14:   # Overlap (highest volume)
        quality_score += 15
    else:                   # Asian/off-hours
        quality_score += 8
    
    # Volatility quality (0-15 points)
    atr = signal_data.get('atr', 0.0012)
    optimal_atr = 0.0015  # Optimal volatility for EURUSD
    atr_deviation = abs(atr - optimal_atr) / optimal_atr
    vol_score = max(0, 15 * (1 - atr_deviation))
    quality_score += vol_score
    
    # Multi-timeframe confirmation (0-20 points)
    if 'htf_15min_trend' in signal_data and 'htf_30min_trend' in signal_data:
        side = signal_data['side']
        htf_15 = signal_data['htf_15min_trend']
        htf_30 = signal_data['htf_30min_trend']
        
        if side == 'long':
            if htf_15 > 0 and htf_30 > 0:
                quality_score += 20  # Strong alignment
            elif htf_15 > 0 or htf_30 > 0:
                quality_score += 10  # Partial alignment
        else:  # short
            if htf_15 < 0 and htf_30 < 0:
                quality_score += 20  # Strong alignment
            elif htf_15 < 0 or htf_30 < 0:
                quality_score += 10  # Partial alignment
    else:
        quality_score += 10  # Default moderate score
    
    return min(quality_score, 100)
```

### 1.2 Advanced Signal Filtering

**Implementation in `train_ranker.py`**:

```python
def apply_enhanced_signal_filters(signals_df):
    """
    Apply advanced filtering to improve signal quality
    """
    # Calculate enhanced quality scores
    signals_df['quality_score'] = signals_df.apply(calculate_enhanced_signal_quality, axis=1)
    
    # Apply quality threshold (top 70% of signals by quality)
    quality_threshold = signals_df['quality_score'].quantile(0.30)
    signals_df = signals_df[signals_df['quality_score'] >= quality_threshold]
    
    # Apply session-specific RR filtering
    def session_rr_filter(row):
        hour = row.name.hour
        min_rr = 2.0  # Default
        
        if 22 <= hour or hour < 8:  # Asian session
            min_rr = 2.5  # Require higher RR in low-volume periods
        elif 8 <= hour < 18:        # London/NY sessions
            min_rr = 1.8  # More permissive in high-volume periods
        
        return row['rr'] >= min_rr
    
    signals_df = signals_df[signals_df.apply(session_rr_filter, axis=1)]
    
    return signals_df
```

### 1.3 Market Regime Optimization

**Current**: Single strategy across all regimes
**Solution**: Regime-specific optimization

```python
def get_regime_specific_settings(regime):
    """
    Return optimized settings for each market regime
    """
    regime_configs = {
        0: {  # Low volatility trending
            'min_probability': 0.65,
            'min_rr': 2.0,
            'max_trades_per_day': 3,
            'cooldown_minutes': 5
        },
        1: {  # Normal volatility
            'min_probability': 0.60,
            'min_rr': 2.2,
            'max_trades_per_day': 4,
            'cooldown_minutes': 8
        },
        2: {  # High volatility/choppy
            'min_probability': 0.70,
            'min_rr': 2.5,
            'max_trades_per_day': 2,
            'cooldown_minutes': 15
        }
    }
    
    return regime_configs.get(regime, regime_configs[1])
```

## Phase 2: Advanced Model Enhancements (Week 3-4)

### 2.1 Meta-Model Improvement

**Current**: Basic meta-model combining base probabilities
**Solution**: Enhanced ensemble with confidence scoring

```python
def train_enhanced_meta_model(features, labels, regime_data):
    """
    Train regime-aware meta-model with confidence estimation
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    
    # Enhanced feature set
    enhanced_features = features.copy()
    
    # Add regime interaction features
    for regime in [0, 1, 2]:
        regime_mask = (regime_data == regime)
        enhanced_features[f'regime_{regime}_prob'] = features['base_prob'] * regime_mask
    
    # Add time-based features
    enhanced_features['hour_sin'] = np.sin(2 * np.pi * features.index.hour / 24)
    enhanced_features['hour_cos'] = np.cos(2 * np.pi * features.index.hour / 24)
    enhanced_features['weekday'] = features.index.weekday
    
    # Train calibrated ensemble
    base_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    # Calibrate for probability reliability
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    calibrated_model.fit(enhanced_features, labels)
    
    return calibrated_model

def get_prediction_confidence(model, features):
    """
    Return both prediction and confidence level
    """
    probabilities = model.predict_proba(features)
    predictions = probabilities[:, 1]
    
    # Calculate confidence based on probability distance from 0.5
    confidence = 2 * np.abs(predictions - 0.5)
    
    return predictions, confidence
```

### 2.2 Dynamic Position Sizing Based on Signal Confidence

**Current**: Fixed position sizing
**Solution**: Confidence-based position sizing

```python
def calculate_dynamic_position_size(signal_confidence, base_risk=0.02):
    """
    Adjust position size based on signal confidence and market conditions
    """
    # Base confidence multiplier
    confidence_multiplier = 0.5 + (signal_confidence * 1.0)  # 0.5x to 1.5x
    
    # Market condition adjustment
    volatility_factor = get_current_volatility_factor()
    vol_multiplier = 1.0 / volatility_factor  # Reduce size in high volatility
    
    # Time-of-day adjustment
    hour = datetime.now().hour
    if 8 <= hour < 18:  # Main trading hours
        time_multiplier = 1.0
    else:  # Off hours
        time_multiplier = 0.7
    
    # Final position size
    adjusted_risk = base_risk * confidence_multiplier * vol_multiplier * time_multiplier
    
    # Apply bounds
    return max(0.01, min(adjusted_risk, 0.04))
```

## Phase 3: Advanced Features (Week 5-6)

### 3.1 Multi-Timeframe Signal Confirmation

**Add to `prepare.py`**:

```python
def add_multitimeframe_features(df_5min):
    """
    Add higher timeframe analysis for signal confirmation
    """
    # 15-minute aggregation
    df_15min = df_5min.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'atr': 'mean'
    }).dropna()
    
    # 30-minute aggregation  
    df_30min = df_5min.resample('30T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'atr': 'mean'
    }).dropna()
    
    # Calculate trend strength
    df_15min['trend_strength'] = (df_15min['close'] - df_15min['close'].shift(8)) / df_15min['atr']
    df_30min['trend_strength'] = (df_30min['close'] - df_30min['close'].shift(4)) / df_30min['atr']
    
    # Add momentum indicators
    df_15min['momentum'] = df_15min['close'].pct_change(periods=4) * 100
    df_30min['momentum'] = df_30min['close'].pct_change(periods=2) * 100
    
    # Merge back to 5-minute data
    df_5min['htf_15min_trend'] = df_15min['trend_strength'].reindex(df_5min.index, method='ffill')
    df_5min['htf_30min_trend'] = df_30min['trend_strength'].reindex(df_5min.index, method='ffill')
    df_5min['htf_15min_momentum'] = df_15min['momentum'].reindex(df_5min.index, method='ffill')
    df_5min['htf_30min_momentum'] = df_30min['momentum'].reindex(df_5min.index, method='ffill')
    
    # Calculate multi-timeframe alignment score
    df_5min['mtf_alignment'] = calculate_mtf_alignment(
        df_5min['htf_15min_trend'], 
        df_5min['htf_30min_trend']
    )
    
    return df_5min

def calculate_mtf_alignment(trend_15min, trend_30min):
    """
    Calculate multi-timeframe trend alignment score (0-1)
    """
    # Both trends in same direction and strong
    strong_alignment = (np.sign(trend_15min) == np.sign(trend_30min)) & \
                      (np.abs(trend_15min) > 1.0) & (np.abs(trend_30min) > 1.0)
    
    # Moderate alignment
    moderate_alignment = (np.sign(trend_15min) == np.sign(trend_30min)) & \
                        ~strong_alignment
    
    # Calculate score
    alignment_score = np.where(strong_alignment, 1.0,
                      np.where(moderate_alignment, 0.6, 0.2))
    
    return alignment_score
```

### 3.2 Advanced Risk Management

```python
class AdvancedRiskManager:
    def __init__(self, max_portfolio_risk=0.06, max_daily_drawdown=0.03):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_daily_drawdown = max_daily_drawdown
        self.open_positions = []
        self.daily_pnl = 0
        self.start_of_day_balance = 10000
    
    def can_take_position(self, new_signal_risk, signal_confidence):
        """
        Advanced position approval logic
        """
        # Check portfolio risk
        current_portfolio_risk = sum(pos['risk'] for pos in self.open_positions)
        if current_portfolio_risk + new_signal_risk > self.max_portfolio_risk:
            return False
        
        # Check daily drawdown
        if self.daily_pnl < -self.max_daily_drawdown * self.start_of_day_balance:
            return False
        
        # Check signal quality threshold
        min_confidence = 0.6 + (current_portfolio_risk / self.max_portfolio_risk) * 0.2
        if signal_confidence < min_confidence:
            return False
        
        # Check correlation with existing positions
        correlation_risk = self.calculate_correlation_risk(new_signal_risk)
        if correlation_risk > 0.7:
            return False
        
        return True
    
    def calculate_correlation_risk(self, new_signal):
        """
        Calculate correlation risk with existing positions
        """
        if not self.open_positions:
            return 0.0
        
        same_direction_risk = sum(
            pos['risk'] for pos in self.open_positions 
            if pos['side'] == new_signal.get('side')
        )
        
        return same_direction_risk / self.max_portfolio_risk
```

## Phase 4: Implementation and Monitoring (Week 7-8)

### 4.1 Configuration Updates

**Update `config.yaml`**:

```yaml
# Enhanced win rate optimization settings
goals:
  win_rate_range: [0.60, 0.75]  # Increased target
  risk_reward_range: [2.0, 3.5]
  trades_per_week_range: [35, 50]  # Maintain current performance

ranker:
  target_trades_per_week: 40
  min_trades_per_week: 35
  max_trades_per_week: 50
  enhanced_filtering: true
  quality_threshold: 0.70
  confidence_based_sizing: true

simulation:
  max_positions: 2
  cooldown_min: 5
  max_weekly_trades: 50
  risk_per_trade: 0.02
  dynamic_sizing: true
  advanced_risk_management: true

signal:
  multi_timeframe_features: true
  enhanced_meta_model: true
  min_confidence_threshold: 0.60
```

### 4.2 Performance Monitoring

```python
def monitor_win_rate_improvements():
    """
    Monitor key metrics for win rate optimization
    """
    metrics = {
        'target_win_rate': 0.65,
        'min_acceptable_win_rate': 0.58,
        'target_trades_per_week': 40,
        'min_trades_per_week': 35,
        'max_acceptable_drawdown': 0.12
    }
    
    # Load recent performance data
    recent_performance = load_recent_results()
    
    # Check win rate stability
    win_rate_std = recent_performance['win_rate'].std()
    if win_rate_std > 0.10:  # More than 10% volatility
        print("⚠️ High win rate volatility detected")
    
    # Check trade frequency
    avg_trades = recent_performance['trades_per_week'].mean()
    if avg_trades < metrics['min_trades_per_week']:
        print("⚠️ Trade frequency below target")
    
    # Overall assessment
    current_win_rate = recent_performance['win_rate'].mean()
    if current_win_rate >= metrics['target_win_rate']:
        print("✅ Win rate optimization successful")
    elif current_win_rate >= metrics['min_acceptable_win_rate']:
        print("⚠️ Win rate improved but needs further optimization")
    else:
        print("❌ Win rate optimization needs adjustment")
    
    return recent_performance
```

## Expected Results by Phase

| Phase | Timeline | Win Rate | Trades/Week | Avg RR | Improvement |
|-------|----------|----------|-------------|--------|-------------|
| Current | - | 45-55% | 42-55 | 3.20 | Baseline |
| Phase 1 | Week 2 | 55-65% | 35-45 | 2.8-3.2 | +10% win rate |
| Phase 2 | Week 4 | 60-70% | 35-45 | 2.5-3.0 | +15% win rate |
| Phase 3 | Week 6 | 62-72% | 35-50 | 2.5-2.8 | +20% win rate |
| Phase 4 | Week 8 | 65-75% | 40-50 | 2.5-2.8 | Target achieved |

## Implementation Commands

### Quick Start:
```bash
# Apply Phase 1 optimizations
python implement_win_rate_improvements.py --apply_phase1

# Run with enhanced configuration
python walkforward.py --run path/to/output --stepback_weeks 4 --optimize --grid fast

# Monitor results
python monitor_win_rate.py
```

### Advanced Implementation:
```bash
# Full pipeline with enhancements
python prepare.py --multi_timeframe --enhanced_features
python train_meta.py --enhanced_model --regime_specific
python train_ranker.py --enhanced_filtering --confidence_scoring
python simulate.py --dynamic_sizing --advanced_risk_management
```

## Key Success Metrics

1. **Win Rate Consistency**: 60-70% with <8% weekly variation
2. **Trade Volume**: Maintain 35+ trades/week  
3. **Risk-Reward**: Keep above 2.5
4. **Max Drawdown**: Control below 12%
5. **Profit Factor**: Target above 2.0

## Risk Mitigation

- **Phase-by-phase implementation** with validation
- **A/B testing** against current system
- **Rollback capability** if metrics deteriorate
- **Continuous monitoring** of all KPIs

## Bottom Line

Your system has excellent trade frequency and RR. By implementing these **signal quality enhancements**, **multi-timeframe confirmation**, and **dynamic risk management**, you can achieve the target 60-70% win rate while maintaining your current strengths.

The key is **gradual implementation** with **continuous monitoring** to ensure each enhancement actually improves performance before proceeding to the next phase.