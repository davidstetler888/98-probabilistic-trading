
# Enhanced Signal Ranking Integration
# Add this code to your train_ranker.py file

def apply_win_rate_enhancements_to_signals(signals_df, target_trades_per_week=40):
    """
    Apply win rate enhancements to generated signals
    """
    try:
        from enhanced_signal_quality import apply_win_rate_enhancements
        
        print("[ranker] Applying win rate enhancements...")
        enhanced_signals = apply_win_rate_enhancements(
            signals_df, 
            target_trades_per_week=target_trades_per_week
        )
        
        print(f"[ranker] Enhanced signals: {len(enhanced_signals)} from {len(signals_df)} original")
        return enhanced_signals
        
    except ImportError:
        print("[ranker] Win rate enhancements not available, using original signals")
        return signals_df
    except Exception as e:
        print(f"[ranker] Error applying enhancements: {e}")
        return signals_df

# Add this call in your main() function after signal generation:
# signals = apply_win_rate_enhancements_to_signals(signals, target)
