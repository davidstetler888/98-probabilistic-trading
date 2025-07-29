#!/usr/bin/env python3
"""
Win Rate Performance Monitor

Monitor win rate, trade frequency, and other KPIs to ensure optimization is working.
"""

import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_results(output_file="output/output.txt"):
    """Analyze results from output.txt"""
    
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
                
            elif "signals/wk" in line:
                # Extract signals per week
                parts = line.split("≈")
                if len(parts) > 1:
                    signals_wk = float(parts[1].split(" ")[0])
                    current_result['signals_per_week'] = signals_wk
            
            elif line.startswith("2024-") and "Win Rate" in line:
                # Week summary line
                if current_result:
                    results.append(current_result.copy())
                    current_result = {}
    
    except FileNotFoundError:
        print(f"❌ Output file not found: {output_file}")
        return []
    
    return results

def analyze_intraday_performance(trades: pd.DataFrame) -> dict:
    """Analyze win rates grouped by hour and weekday.

    Parameters
    ----------
    trades : pandas.DataFrame
        DataFrame returned by ``simulate.simulate_df`` with
        ``return_trades=True`` containing ``timestamp`` and ``profit`` columns.

    Returns
    -------
    dict
        Dictionary with ``by_hour`` and ``by_weekday`` win rate Series.
    """

    if trades.empty:
        empty = pd.Series(dtype=float)
        return {"by_hour": empty, "by_weekday": empty}

    df = trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df["hour"] = df["timestamp"].dt.hour
    by_hour = df.groupby("hour").apply(lambda g: (g["profit"] > 0).mean()).sort_index()

    df["weekday"] = df["timestamp"].dt.day_name()
    by_weekday = df.groupby("weekday").apply(lambda g: (g["profit"] > 0).mean())
    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    by_weekday = by_weekday.reindex(order).dropna()

    print("\n⌛ WIN RATE BY HOUR:")
    for h, wr in by_hour.items():
        print(f"   {h:02d}:00  {wr*100:.1f}%")

    print("\n📅 WIN RATE BY WEEKDAY:")
    for day, wr in by_weekday.items():
        print(f"   {day:<9} {wr*100:.1f}%")

    return {"by_hour": by_hour, "by_weekday": by_weekday}

def generate_performance_report(results):
    """Generate performance analysis report"""
    
    if not results:
        print("❌ No results to analyze")
        return
    
    print("\n" + "="*60)
    print("📊 WIN RATE OPTIMIZATION PERFORMANCE REPORT")
    print("="*60)
    
    # Calculate statistics
    df = pd.DataFrame(results)
    
    if not df.empty:
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   • Total Periods Analyzed: {len(df)}")
        print(f"   • Average Win Rate: {df['win_rate'].mean():.1f}% (σ={df['win_rate'].std():.1f}%)")
        print(f"   • Average Trades/Period: {df['total_trades'].mean():.1f}")
        print(f"   • Average RR: {df['avg_rr'].mean():.2f}")
        
        # Stability analysis
        win_rate_stability = df['win_rate'].std()
        if win_rate_stability < 10:
            stability_rating = "✅ EXCELLENT"
        elif win_rate_stability < 20:
            stability_rating = "✅ GOOD"
        elif win_rate_stability < 30:
            stability_rating = "⚠️ MODERATE"
        else:
            stability_rating = "❌ POOR"
        
        print(f"\n🎯 WIN RATE STABILITY: {stability_rating}")
        print(f"   • Standard Deviation: {win_rate_stability:.1f}%")
        print(f"   • Min Win Rate: {df['win_rate'].min():.1f}%")
        print(f"   • Max Win Rate: {df['win_rate'].max():.1f}%")
        
        # Trade frequency analysis
        avg_trades = df['total_trades'].mean()
        if avg_trades >= 25:
            frequency_rating = "✅ TARGET ACHIEVED"
        elif avg_trades >= 15:
            frequency_rating = "✅ IMPROVED"
        elif avg_trades >= 10:
            frequency_rating = "⚠️ MODERATE IMPROVEMENT"
        else:
            frequency_rating = "❌ INSUFFICIENT"
        
        print(f"\n📊 TRADE FREQUENCY: {frequency_rating}")
        print(f"   • Current Average: {avg_trades:.1f} trades/period")
        print(f"   • Target Range: 25-50 trades/week")
        
        # Overall assessment
        print(f"\n🏆 OVERALL ASSESSMENT:")
        if df['win_rate'].mean() >= 60 and avg_trades >= 20 and win_rate_stability < 15:
            print("   ✅ OPTIMIZATION SUCCESSFUL!")
            print("   ✅ Win rate stable and within target range")
            print("   ✅ Trade frequency significantly improved")
        elif df['win_rate'].mean() >= 55 and avg_trades >= 15:
            print("   ⚠️ OPTIMIZATION PARTIALLY SUCCESSFUL")
            print("   ⚠️ Continue with Phase 2 improvements")
        else:
            print("   ❌ OPTIMIZATION NEEDS ADJUSTMENT")
            print("   ❌ Consider reverting to previous configuration")

def main():
    print("🔍 Analyzing win rate optimization results...")
    results = analyze_results()
    generate_performance_report(results)

if __name__ == "__main__":
    main()
