#!/usr/bin/env python3
"""
Enhanced Win Rate Performance Monitor

Monitor win rate improvements and provide actionable insights.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

def analyze_win_rate_performance(output_file="output/output.txt"):
    """Analyze win rate performance from walkforward results"""
    
    results = []
    
    try:
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        current_result = {}
        for line in lines:
            line = line.strip()
            
            if "Total Trades:" in line:
                trades = int(line.split(": ")[1])
                current_result['total_trades'] = trades
                
            elif "Win Rate:" in line:
                win_rate = float(line.split(": ")[1].replace("%", ""))
                current_result['win_rate'] = win_rate
                
            elif "Average RR:" in line:
                avg_rr = float(line.split(": ")[1])
                current_result['avg_rr'] = avg_rr
                
            elif "Profit Factor:" in line:
                pf = line.split(": ")[1]
                if pf == "inf":
                    current_result['profit_factor'] = float('inf')
                else:
                    current_result['profit_factor'] = float(pf)
                
            elif line.startswith("2024-") and "Win Rate" in line:
                # Week summary line
                if current_result:
                    results.append(current_result.copy())
                    current_result = {}
    
    except FileNotFoundError:
        print(f"❌ Output file not found: {output_file}")
        return []
    
    return results

def generate_win_rate_report(results):
    """Generate comprehensive win rate analysis report"""
    
    if not results:
        print("❌ No results to analyze")
        return
    
    print("\n" + "="*70)
    print("📊 WIN RATE OPTIMIZATION PERFORMANCE REPORT")
    print("="*70)
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        current_win_rate = df['win_rate'].mean()
        win_rate_std = df['win_rate'].std()
        avg_trades = df['total_trades'].mean()
        avg_rr = df['avg_rr'].mean()
        
        print(f"\n📈 PERFORMANCE SUMMARY:")
        print(f"   • Periods Analyzed: {len(df)}")
        print(f"   • Average Win Rate: {current_win_rate:.1f}% (σ={win_rate_std:.1f}%)")
        print(f"   • Average Trades/Period: {avg_trades:.1f}")
        print(f"   • Average RR: {avg_rr:.2f}")
        
        # Win rate assessment
        print(f"\n🎯 WIN RATE ANALYSIS:")
        if current_win_rate >= 65:
            rating = "✅ EXCELLENT"
            recommendation = "Target achieved! Consider further optimization."
        elif current_win_rate >= 60:
            rating = "✅ GOOD"
            recommendation = "Close to target. Minor adjustments may help."
        elif current_win_rate >= 55:
            rating = "⚠️ ACCEPTABLE"
            recommendation = "Improvement needed. Apply Phase 2 enhancements."
        else:
            rating = "❌ NEEDS WORK"
            recommendation = "Significant improvement required. Check configuration."
        
        print(f"   • Rating: {rating}")
        print(f"   • Recommendation: {recommendation}")
        
        # Stability analysis
        print(f"\n📊 WIN RATE STABILITY:")
        if win_rate_std < 8:
            stability = "✅ STABLE"
        elif win_rate_std < 15:
            stability = "⚠️ MODERATE"
        else:
            stability = "❌ VOLATILE"
        
        print(f"   • Stability: {stability} (σ={win_rate_std:.1f}%)")
        print(f"   • Min Win Rate: {df['win_rate'].min():.1f}%")
        print(f"   • Max Win Rate: {df['win_rate'].max():.1f}%")
        
        # Trade frequency analysis
        print(f"\n📊 TRADE FREQUENCY:")
        if avg_trades >= 35:
            freq_rating = "✅ TARGET MET"
        elif avg_trades >= 25:
            freq_rating = "⚠️ CLOSE TO TARGET"
        else:
            freq_rating = "❌ BELOW TARGET"
        
        print(f"   • Rating: {freq_rating}")
        print(f"   • Current: {avg_trades:.1f} trades/period")
        print(f"   • Target: 35+ trades/period")
        
        # Overall assessment
        print(f"\n🏆 OVERALL ASSESSMENT:")
        
        success_score = 0
        if current_win_rate >= 60: success_score += 40
        elif current_win_rate >= 55: success_score += 20
        
        if win_rate_std < 10: success_score += 20
        elif win_rate_std < 15: success_score += 10
        
        if avg_trades >= 35: success_score += 25
        elif avg_trades >= 25: success_score += 15
        
        if avg_rr >= 2.5: success_score += 15
        elif avg_rr >= 2.0: success_score += 10
        
        if success_score >= 80:
            overall = "✅ OPTIMIZATION SUCCESSFUL!"
            next_steps = "System performing excellently. Monitor and maintain."
        elif success_score >= 60:
            overall = "⚠️ PARTIAL SUCCESS"
            next_steps = "Good progress. Apply additional enhancements."
        else:
            overall = "❌ NEEDS IMPROVEMENT"
            next_steps = "Review configuration and apply Phase 1 fixes."
        
        print(f"   • Result: {overall}")
        print(f"   • Score: {success_score}/100")
        print(f"   • Next Steps: {next_steps}")

def main():
    print("🔍 Analyzing win rate optimization results...")
    results = analyze_win_rate_performance()
    generate_win_rate_report(results)

if __name__ == "__main__":
    main()
