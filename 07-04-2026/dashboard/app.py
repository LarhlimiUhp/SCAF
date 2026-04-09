import streamlit as st
import pandas as pd


def run_dashboard(backtest_results=None):
    st.title('SCAF-LS Backtest Dashboard')
    if backtest_results is None:
        st.write('Aucun résultat disponible. Exécutez d’abord le backtest.')
        return

    backtest = backtest_results.get('backtest')
    if backtest is None or backtest.empty:
        st.warning('Aucun backtest disponible. Vérifiez les données et ré-exécutez la stratégie.')
        return

    st.subheader('Equity Curve')
    st.line_chart(backtest['equity'])

    st.subheader('Strategy vs Benchmark')
    st.line_chart(backtest[['equity', 'benchmark']])

    st.subheader('Position and Signal')
    st.line_chart(backtest[['position', 'signal']])

    st.subheader('Performance Summary')
    st.metric('Final equity', f"${backtest['equity'].iloc[-1]:,.0f}")
    max_dd = float((backtest['equity'] / backtest['equity'].cummax() - 1).min())
    st.metric('Max drawdown', f"{max_dd:.2%}")
    if 'summary' in backtest_results:
        st.json(backtest_results['summary'])

    st.subheader('Fold CV Scores')
    for idx, fold in enumerate(backtest_results.get('cv_scores', []), start=1):
        st.write(f'Fold {idx}')
        st.json(fold)

    st.subheader('Raw Backtest Data')
    st.dataframe(backtest.reset_index().tail(50))
