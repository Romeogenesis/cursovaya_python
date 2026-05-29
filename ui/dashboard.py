from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.data_loader import DataLoader
from core.quality_assessor import (
    QualityAssessor,
    MinMaxNormalization,
    ReferenceNormalization,
    DefaultWeightStrategy,
    CustomWeightStrategy,
    EquipmentSpecificWeightStrategy
)
from core.forecast_engine import (
    ForecastEngine,
    LinearRegressionStrategy,
    PolynomialRegressionStrategy
)
from core.resource_optimizer import ResourceOptimizer
from utils.config import (
    DEFAULT_WEIGHTS,
    EQUIPMENT_PROFILES,
    OPTIMIZATION_DEFAULTS,
    PLOTLY_TEMPLATE
)


def render_data_tab(data_loader: DataLoader):
    st.header("📊 Данные")
    st.subheader("Загрузка данных")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Загрузить CSV файл",
            type=['csv'],
            help="Файл должен содержать колонки: avg_equipment_price, measurement_accuracy, is_digital_display, operating_temperature, weight, monthly_equipment_sales"
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if data_loader.load_from_dict(df.to_dict(orient='list')):
                    st.success("✅ Данные загружены!")
            except Exception as e:
                st.error(f"Ошибка: {e}")
    
    with col2:
        if st.button("🎲 Тестовые данные"):
            from core.data_loader import create_sample_dataset
            import tempfile, os
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    temp_path = f.name
                    create_sample_dataset(temp_path, n_samples=100)
                if data_loader.load_csv(temp_path):
                    st.success("✅ Тестовые данные сгенерированы!")
            except Exception as e:
                st.error(f"Ошибка: {e}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    try: os.unlink(temp_path)
                    except: pass
    
    st.subheader("Ручной ввод")
    with st.expander("➕ Добавить образец"):
        with st.form("manual_form"):
            c1, c2 = st.columns(2)
            with c1:
                price = st.number_input("Цена (₽)", 0.0, value=150.0)
                accuracy = st.number_input("Точность", 0.00001, value=0.05, format="%.5f")
                digital = st.checkbox("Цифровой дисплей", True)
            with c2:
                temp = st.number_input("Температура (°C)", 0.0, value=80.0)
                weight = st.number_input("Вес (кг)", 0.0, value=15.0)
                sales = st.number_input("Продажи/мес", 0, value=673)
            if st.form_submit_button("Добавить"):
                new = {
                    'avg_equipment_price': [price],
                    'measurement_accuracy': [accuracy],
                    'is_digital_display': [1 if digital else 0],
                    'operating_temperature': [temp],
                    'weight': [weight],
                    'monthly_equipment_sales': [sales]
                }
                if data_loader.load_from_dict(new):
                    st.success("✅ Добавлено!")
                else:
                    st.error(f"Ошибка: {data_loader.last_error}")
    
    st.subheader("Просмотр")
    data = data_loader.data
    if data is not None and not data.empty:
        st.dataframe(data.head(20), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Записей", len(data))
        c2.metric("Колонок", len(data.columns))
        c3.metric("Числовых", len(data.select_dtypes(include=[np.number]).columns))
        if st.checkbox("📊 Статистика"):
            st.write(data.describe())
    else:
        st.warning("⚠️ Нет данных. Загрузите файл.")
    
    if data is not None and not data.empty:
        st.download_button("📥 Скачать CSV", data.to_csv(index=False).encode('utf-8'), 'data.csv', 'text/csv')


def render_technical_level_tab(data_loader: DataLoader):
    st.header("🎯 Технический уровень")
    data = data_loader.data
    if data is None or data.empty:
        st.warning("⚠️ Загрузите данные на вкладке 'Данные'")
        return
    
    st.sidebar.subheader("⚙️ Параметры")
    equip_type = st.sidebar.selectbox("Тип оборудования", list(EQUIPMENT_PROFILES.keys()), format_func=lambda x: EQUIPMENT_PROFILES[x]['name'])
    norm_method = st.sidebar.radio("Нормализация", ["Min-Max", "По эталону"], 0)
    weight_strat = st.sidebar.radio("Веса", ["По умолчанию", "Для типа", "Свои"], 0)
    
    assessor = QualityAssessor()
    if norm_method == "Min-Max":
        assessor.set_normalization_strategy(MinMaxNormalization())
    else:
        ref = assessor.get_reference_profile(equip_type)
        if ref: assessor.set_normalization_strategy(ReferenceNormalization(ref))
    
    if weight_strat == "По умолчанию":
        assessor.set_weight_strategy(DefaultWeightStrategy())
    elif weight_strat == "Для типа":
        assessor.set_weight_strategy(EquipmentSpecificWeightStrategy())
    else:
        st.sidebar.subheader("🎚 Настройка весов")
        cw, total = {}, 0
        for m, dw in DEFAULT_WEIGHTS.items():
            w = st.sidebar.slider(f"{m}", 0.0, 1.0, dw, 0.05)
            cw[m] = w; total += w
        if abs(total - 1.0) <= 0.01:
            try: assessor.set_weight_strategy(CustomWeightStrategy(cw))
            except ValueError as e: st.sidebar.error(str(e))
        else:
            st.sidebar.warning(f"Сумма весов: {total:.2f} (должно быть 1.0)")
    
    df_res, err = assessor.calculate_technical_level(data, equip_type)
    if err: st.error(f"❌ Ошибка: {err}"); return
    
    st.subheader("📈 Результаты")
    c1, c2, c3 = st.columns(3)
    c1.metric("Средний", f"{df_res['technical_level'].mean():.3f}")
    c2.metric("Макс", f"{df_res['technical_level'].max():.3f}")
    c3.metric("Мин", f"{df_res['technical_level'].min():.3f}")
    
    st.subheader("📊 Графики")
    vc1, vc2 = st.columns(2)
    
    with vc1:
        st.markdown("#### 🔵 Радиальная диаграмма")
        ref = assessor.get_reference_profile(equip_type)
        if ref and any(c.startswith('norm_') for c in df_res.columns):
            metrics = ['avg_equipment_price', 'measurement_accuracy', 'is_digital_display', 'operating_temperature', 'weight']
            idx = df_res['technical_level'].idxmax()
            sample = df_res.loc[idx]
            cats, vals, refs = [], [], []
            ru = {'avg_equipment_price': 'Цена', 'measurement_accuracy': 'Точность', 'is_digital_display': 'Дисплей', 'operating_temperature': 'Темп.', 'weight': 'Вес'}
            for m in metrics:
                nc = f'norm_{m}'
                if nc in df_res.columns:
                    cats.append(ru.get(m, m))
                    vals.append(sample.get(nc, 0))
                    refs.append(1.0)
            if cats:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill='toself', name='Образец'))
                fig.add_trace(go.Scatterpolar(r=refs, theta=cats, fill='toself', name='Эталон', line=dict(dash='dash')))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, template=PLOTLY_TEMPLATE, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("ℹ️ Нет данных для диаграммы")
        else: st.info("ℹ️ Нет данных для диаграммы")
    
    with vc2:
        st.markdown("#### 📊 Столбчатая (по убыванию)")
        df_plot = df_res.sort_values('technical_level', ascending=False).head(20).copy()
        df_plot['label'] = [f"#{i+1}" for i in range(len(df_plot))]
        if not df_plot.empty and 'technical_level' in df_plot.columns:
            fig = px.bar(df_plot, x='label', y='technical_level', color='technical_level', color_continuous_scale='Viridis', labels={'technical_level': 'Уровень', 'label': 'Образец'})
            fig.update_layout(xaxis_title="Образец", yaxis_title="Тех. уровень", template=PLOTLY_TEMPLATE, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else: st.warning("⚠️ Нет данных")
    
    if st.checkbox("📋 Детальная таблица"):
        cols = [c for c in df_res.columns if not c.startswith('norm_')]
        st.dataframe(df_res[cols], use_container_width=True)


def render_forecast_tab(data_loader: DataLoader):
    st.header("📈 Прогноз")
    data = data_loader.data
    if data is None or data.empty: st.warning("⚠️ Загрузите данные"); return
    
    engine = ForecastEngine()
    st.subheader("🔗 Корреляции")
    target = st.selectbox("Целевая переменная", ['monthly_equipment_sales', 'avg_equipment_price'], 0)
    if target not in data.columns: st.warning(f"❌ Колонка '{target}' не найдена"); return
    
    corr = engine.analyze_correlations(data, target)
    if not corr.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.imshow(corr, text_auto='.2f', aspect='auto', color_continuous_scale='RdBu_r')
            fig.update_layout(title='Матрица корреляций', height=400)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sig = engine.get_significant_features(data, target)
            if sig:
                feat = st.selectbox("Признак для scatter", sig)
                if feat in data.columns:
                    fig = px.scatter(data, x=feat, y=target, trendline='ols', title=f'{feat} vs {target}')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else: st.info("ℹ️ Нет значимых корреляций (>0.3)")
    
    st.subheader("🤖 Обучение модели")
    model = st.radio("Тип", ["Линейная", "Полиномиальная (deg=2)"], 0)
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if target in num_cols: num_cols.remove(target)
    feats = st.multiselect("Признаки", num_cols, default=num_cols[:3] if len(num_cols)>=3 else num_cols)
    
    if st.button("🚀 Обучить"):
        if len(feats) < 1: st.error("Выберите хотя бы один признак")
        else:
            engine.set_strategy(PolynomialRegressionStrategy(degree=2) if "Полиномиальная" in model else LinearRegressionStrategy())
            metrics, err = engine.fit(data, feats, target)
            if err: st.error(f"❌ {err}")
            else:
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("R² train", f"{metrics['train_r2']:.4f}")
                c2.metric("R² test", f"{metrics['test_r2']:.4f}")
                c3.metric("MSE test", f"{metrics['test_mse']:.4f}")
                c4.metric("Переобучение", "⚠️ Да" if metrics['is_overfitting'] else "✅ Нет")
                if st.checkbox("Кросс-валидация"):
                    cv = engine.cross_validate(data, feats, target)
                    if cv:
                        st.write(f"**R²:** {cv['cv_mean_r2']:.4f} ± {cv['cv_std_r2']:.4f}")
                        n = min(len(cv['cv_scores']), 5)
                        fig = px.bar(x=[f'Fold {i+1}' for i in range(n)], y=cv['cv_scores'][:n], labels={'x':'Fold','y':'R²'}, title='R² по фолдам')
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🔮 Прогноз")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("#### Точечный")
        inp = {}
        for f in feats:
            if f in data.columns and pd.api.types.is_numeric_dtype(data[f]):
                inp[f] = st.number_input(f, value=float(data[f].median()), key=f"p_{f}")
        if st.button("Прогнозировать"):
            if engine.feature_names:
                pred, err = engine.predict(inp)
                if err: st.error(f"❌ {err}")
                else: st.success(f"🎯 {target}: **{pred:.2f}**")
            else: st.warning("⚠️ Сначала обучите модель")
    
    with pc2:
        st.markdown("#### Интервальный анализ")
        if engine.feature_names and feats:
            var_f = st.selectbox("Варируемый признак", feats, key="vf")
            fixed = inp.copy()
            if var_f in data.columns and pd.api.types.is_numeric_dtype(data[var_f]):
                mn = st.number_input(f"Мин {var_f}", value=float(data[var_f].min()), key=f"min_{var_f}")
                mx = st.number_input(f"Макс {var_f}", value=float(data[var_f].max()), key=f"max_{var_f}")
                if st.button("Построить"):
                    res, err = engine.interval_analysis(data, fixed, var_f, mn, mx)
                    if err or res is None or res.empty: st.error(f"❌ {err or 'Пустой результат'}")
                    else:
                        pc = f'predicted_{target}'
                        if pc in res.columns:
                            fig = px.line(res, x=var_f, y=pc, title=f'{target} от {var_f}', markers=True)
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else: st.error(f"❌ Колонка '{pc}' не найдена")
        else: st.info("ℹ️ Обучите модель")


def render_resources_tab(data_loader: DataLoader):
    st.header("🔧 Оптимизация закупок")
    data = data_loader.data
    if data is None or data.empty: st.warning("⚠️ Загрузите данные"); return
    
    optimizer = ResourceOptimizer()
    st.subheader("⚙️ Параметры")
    c1, c2, c3 = st.columns(3)
    budget = c1.number_input("Бюджет (₽)", 0, value=int(OPTIMIZATION_DEFAULTS['budget']), step=10000)
    min_dem = c2.number_input("Мин. спрос", 0, value=int(OPTIMIZATION_DEFAULTS['min_demand']), step=1)
    wh_cap = c3.number_input("Ёмкость склада", 1, value=int(OPTIMIZATION_DEFAULTS['warehouse_capacity']), step=10)
    
    if st.button("🚀 Оптимизировать"):
        res, err = optimizer.optimize_procurement(data, budget=budget, min_demand=min_dem, warehouse_capacity=wh_cap)
        if err: st.error(f"❌ {err}")
        elif res is None or res.empty: st.warning("⚠️ Нет решения")
        else:
            st.subheader("✅ Результаты")
            summ = optimizer.get_optimization_summary()
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Оптимальные затраты", f"{summ.get('total_optimized_cost',0):,.0f} ₽")
            c2.metric("Наивный план", f"{summ.get('total_naive_cost',0):,.0f} ₽")
            sav = summ.get('savings',0); nc = summ.get('total_naive_cost',1)
            c3.metric("Экономия", f"{sav:,.0f} ₽", delta=f"{sav/nc*100:.1f}%" if nc>0 else "0%")
            c4.metric("Бюджет", f"{summ.get('budget_used',0):.1f}%")
            
            vc1, vc2 = st.columns(2)
            with vc1:
                st.markdown("#### План vs Оптимум")
                if all(c in res.columns for c in ['equipment_type','optimal_quantity','naive_quantity']):
                    cmp = res.melt(id_vars=['equipment_type'], value_vars=['optimal_quantity','naive_quantity'], var_name='plan', value_name='qty')
                    fig = px.bar(cmp, x='equipment_type', y='qty', color='plan', barmode='group', color_discrete_map={'optimal_quantity':'#2E86AB','naive_quantity':'#A23B72'})
                    fig.update_layout(height=400, template=PLOTLY_TEMPLATE)
                    st.plotly_chart(fig, use_container_width=True)
            with vc2:
                st.markdown("#### Структура затрат")
                if 'equipment_type' in res.columns and 'total_cost' in res.columns:
                    fig = px.pie(res, values='total_cost', names='equipment_type', title='Затраты')
                    fig.update_layout(height=400, template=PLOTLY_TEMPLATE)
                    st.plotly_chart(fig, use_container_width=True)
            
            if st.checkbox("📋 Таблица"):
                st.dataframe(res, use_container_width=True)
            st.download_button("📥 Скачать CSV", res.to_csv(index=False).encode('utf-8'), 'optimization.csv', 'text/csv')
    
    st.subheader("📊 Анализ чувствительности")
    param = st.selectbox("Параметр", ['budget','min_demand','warehouse_capacity'], format_func=lambda x: {'budget':'Бюджет','min_demand':'Спрос','warehouse_capacity':'Склад'}[x])
    if st.button("Анализ"):
        sens, err = optimizer.sensitivity_analysis(data, parameter=param)
        if err or sens is None or sens.empty: st.error(f"❌ {err or 'Пустой результат'}")
        else:
            if param in sens.columns and 'total_cost' in sens.columns:
                fig = px.line(sens, x=param, y='total_cost', title=f'Затраты от {param}', markers=True)
                fig.update_layout(height=400, template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)


def render_transport_tab():
    st.header("🚚 Оптимизация")
    st.markdown("""
    **Распределение метрологического оборудования от заводов M1–M3 к лабораториям Lab1–Lab4**
    
    Цель: минимизация затрат на производство и логистику при соблюдении мощностей заводов и полного удовлетворения спроса лабораторий.
    """)
    
    optimizer = ResourceOptimizer()
    
    tab1, tab2 = st.tabs(["📊 Быстрый ввод (по варианту 29)", "⚙️ Произвольные данные"])
    
    with tab1:
        st.subheader("Данные по варианту №29")
        
        default_costs = np.array([
            [250, 280, 320, 350],
            [300, 260, 290, 330],
            [360, 320, 270, 310]
        ])
        
        default_supply = np.array([400, 350, 300])
        default_demand = np.array([200, 250, 300, 150])
        
        st.write("**Матрица затрат (тыс. руб./ед.):**")
        costs_df = pd.DataFrame(
            default_costs,
            index=['M1', 'M2', 'M3'],
            columns=['Lab1', 'Lab2', 'Lab3', 'Lab4']
        )
        st.dataframe(costs_df.style.format("{:.0f}"), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Мощности заводов (ед./год):**")
            supply_df = pd.DataFrame({
                'Завод': ['M1', 'M2', 'M3'],
                'Мощность': default_supply
            })
            st.dataframe(supply_df, use_container_width=True)
        
        with col2:
            st.write("**Спрос лабораторий (ед./год):**")
            demand_df = pd.DataFrame({
                'Лаборатория': ['Lab1', 'Lab2', 'Lab3', 'Lab4'],
                'Спрос': default_demand
            })
            st.dataframe(demand_df, use_container_width=True)
        
        if st.button("🚀 Решить транспортную задачу (Вариант 29)", type="primary", key="solve_default"):
            solve_and_display_transport(optimizer, default_costs, default_supply, default_demand)
    
    with tab2:
        st.subheader("Произвольные данные")
        
        n_plants = st.number_input("Количество заводов", min_value=2, max_value=5, value=3, key="n_plants")
        n_labs = st.number_input("Количество лабораторий", min_value=2, max_value=6, value=4, key="n_labs")
        
        st.write("**Матрица затрат (тыс. руб./ед.):**")
        costs_input = []
        for i in range(n_plants):
            row = []
            cols = st.columns(n_labs)
            for j in range(n_labs):
                val = cols[j].number_input(
                    f"M{i+1}→Lab{j+1}",
                    min_value=0.0,
                    value=float(200 + i*50 + j*30),
                    key=f"cost_{i}_{j}"
                )
                row.append(val)
            costs_input.append(row)
        
        costs_custom = np.array(costs_input)
        
        st.write("**Мощности заводов:**")
        supply_input = []
        for i in range(n_plants):
            val = st.number_input(
                f"Завод M{i+1}",
                min_value=0,
                value=300 + i*50,
                key=f"supply_{i}"
            )
            supply_input.append(val)
        
        supply_custom = np.array(supply_input)
        
        st.write("**Спрос лабораторий:**")
        demand_input = []
        for j in range(n_labs):
            val = st.number_input(
                f"Лаборатория Lab{j+1}",
                min_value=0,
                value=200 + j*50,
                key=f"demand_{j}"
            )
            demand_input.append(val)
        
        demand_custom = np.array(demand_input)
        
        if st.button("🚀 Решить транспортную задачу", type="primary", key="solve_custom"):
            solve_and_display_transport(optimizer, costs_custom, supply_custom, demand_custom)


def solve_and_display_transport(optimizer: ResourceOptimizer, costs: np.ndarray, 
                                 supply: np.ndarray, demand: np.ndarray):
    
    with st.spinner("⏳ Решение транспортной задачи..."):
        result_df, error = optimizer.solve_transport_problem(costs, supply, demand)
        
        if error:
            st.error(f"❌ Ошибка: {error}")
            return
        
        st.success("✅ Задача успешно решена!")
        
        st.subheader("📈 Результаты оптимизации")
        opt_result = optimizer.get_optimization_summary()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "📦 Минимальные затраты",
                f"{opt_result.get('total_cost', 0):,} тыс. руб.",
                delta="Оптимально"
            )
        with col2:
            total_supply = sum(supply)
            used_supply = sum(result_df['Объём_поставки']) if 'Объём_поставки' in result_df.columns else 0
            st.metric(
                "🏭 Использовано мощностей",
                f"{used_supply} из {total_supply} ед.",
                delta=f"{used_supply/total_supply*100:.1f}%" if total_supply > 0 else "0%"
            )
        with col3:
            total_demand = sum(demand)
            st.metric(
                "🔬 Удовлетворён спрос",
                f"{total_demand} ед.",
                delta="100%"
            )
        
        st.divider()
        
        st.subheader("📋 Оптимальный план перевозок")
        
        plants = ['M1', 'M2', 'M3'][:len(supply)]
        labs = ['Lab1', 'Lab2', 'Lab3', 'Lab4'][:len(demand)]
        
        transport_matrix = np.zeros((len(plants), len(labs)))
        for _, row in result_df.iterrows():
            i = plants.index(row['Завод'])
            j = labs.index(row['Лаборатория'])
            transport_matrix[i, j] = row['Объём_поставки']
        
        transport_df = pd.DataFrame(
            transport_matrix,
            index=plants,
            columns=labs
        )
        
        st.dataframe(transport_df, use_container_width=True)
        
        st.write("**Детализация перевозок:**")
        st.dataframe(
            result_df.style.format({
                'Стоимость_ед': '{:.0f}',
                'Затраты': '{:.0f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Структура затрат")
            
            if not result_df.empty:
                fig_pie = px.pie(
                    result_df,
                    values='Затраты',
                    names='Завод',
                    title='Распределение затрат по заводам',
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("📦 Объёмы поставок")
            
            if not result_df.empty:
                fig_bar = px.bar(
                    result_df,
                    x='Лаборатория',
                    y='Объём_поставки',
                    color='Завод',
                    title='Распределение поставок по лабораториям',
                    barmode='stack'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        st.subheader("🗺️ Тепловая карта оптимальных перевозок")
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=transport_matrix,
            x=labs,
            y=plants,
            colorscale='YlGnBu',
            text=transport_matrix.astype(int),
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Ед.")
        ))
        
        fig_heatmap.update_layout(
            title='Матрица оптимальных перевозок',
            xaxis_title='Лаборатории',
            yaxis_title='Заводы'
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.divider()
        st.subheader("📥 Экспорт результатов")
        
        csv_data = result_df.to_csv(index=False, sep=';', decimal=',')
        st.download_button(
            label="📥 Скачать результаты (.csv)",
            data=csv_data,
            file_name="transport_optimization_result.csv",
            mime="text/csv"
        )


def main():
    st.set_page_config(page_title="🏭 Метрологическое оборудование", page_icon="📊", layout="wide")
    st.title("🏭 ИАС: Производитель метрологического оборудования")
    
    data_loader = DataLoader()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Данные", 
        "🎯 Тех. уровень", 
        "📈 Прогноз", 
        "🔧 Ресурсы",
        "🚚 Транспортная задача"
    ])
    
    with tab1: render_data_tab(data_loader)
    with tab2: render_technical_level_tab(data_loader)
    with tab3: render_forecast_tab(data_loader)
    with tab4: render_resources_tab(data_loader)
    with tab5: render_transport_tab()
    
    st.markdown("---")
    st.caption("Курсовая работа • Вариант №29 • РУТ (МИИТ) • 2026")


if __name__ == "__main__":
    main()