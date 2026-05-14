"""
Streamlit dashboard for Metrological Equipment Analysis System.
Implements the View component of MVC pattern.
"""
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
    """Render the Data tab with data upload and preview."""
    st.header("📊 Данные")
    
    # File upload section
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
                if data_loader.load_from_dict(df.to_dict()):
                    st.success("Данные успешно загружены!")
            except Exception as e:
                st.error(f"Ошибка загрузки файла: {e}")
    
    with col2:
        if st.button("Сгенерировать тестовые данные"):
            from core.data_loader import create_sample_dataset
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                temp_path = f.name
                if create_sample_dataset(temp_path, n_samples=100):
                    if data_loader.load_csv(temp_path):
                        st.success("Тестовые данные сгенерированы!")
                    os.unlink(temp_path)
    
    # Manual data entry
    st.subheader("Ручной ввод данных")
    
    with st.expander("Добавить образец оборудования"):
        with st.form("manual_entry_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                price = st.number_input("Цена оборудования (руб)", min_value=0.0, value=50000.0)
                accuracy = st.number_input("Точность измерений", min_value=0.00001, value=0.001, format="%.6f")
                digital = st.checkbox("Цифровой дисплей", value=True)
            
            with col_b:
                temp_range = st.number_input("Диапазон температур (°C)", min_value=0.0, value=50.0)
                weight = st.number_input("Вес (кг)", min_value=0.0, value=5.0)
                sales = st.number_input("Месячные продажи (шт)", min_value=0, value=100)
            
            submitted = st.form_submit_button("Добавить")
            
            if submitted:
                new_data = {
                    'price': [price],
                    'accuracy': [accuracy],
                    'digital_display': [1 if digital else 0],
                    'temperature_range': [temp_range],
                    'weight': [weight],
                    'sales': [sales]
                }
                
                if data_loader.load_from_dict(new_data):
                    st.success("Образец добавлен!")
                else:
                    st.error(f"Ошибка: {data_loader.last_error}")
    
    # Data preview
    st.subheader("Просмотр данных")
    
    data = data_loader.data
    
    if data is not None and not data.empty:
        st.dataframe(data.head(20), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Всего записей", len(data))
        with col2:
            st.metric("Колонок", len(data.columns))
        with col3:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            st.metric("Числовых колонок", len(numeric_cols))
        
        # Statistics
        if st.checkbox("Показать статистику"):
            st.write(data.describe())
    else:
        st.warning("Данные не загружены. Загрузите файл или используйте тестовые данные.")
    
    # Download button
    if data is not None and not data.empty:
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Скачать данные (CSV)",
            data=csv,
            file_name='metrological_data.csv',
            mime='text/csv'
        )


def render_technical_level_tab(data_loader: DataLoader):
    """Render the Technical Level tab with assessment and visualization."""
    st.header("🎯 Оценка технического уровня")
    
    data = data_loader.data
    
    if data is None or data.empty:
        st.warning("Сначала загрузите данные на вкладке 'Данные'")
        return
    
    # Settings sidebar
    st.sidebar.subheader("Параметры оценки")
    
    equipment_type = st.sidebar.selectbox(
        "Тип оборудования",
        options=list(EQUIPMENT_PROFILES.keys()),
        format_func=lambda x: EQUIPMENT_PROFILES[x]['name']
    )
    
    normalization_method = st.sidebar.radio(
        "Метод нормализации",
        options=["Min-Max", "По эталону"],
        index=0
    )
    
    weight_strategy = st.sidebar.radio(
        "Стратегия весов",
        options=["По умолчанию", "Для типа оборудования", "Пользовательская"],
        index=0
    )
    
    # Initialize assessor with selected strategies
    assessor = QualityAssessor()
    
    if normalization_method == "Min-Max":
        assessor.set_normalization_strategy(MinMaxNormalization())
    else:
        ref_profile = assessor.get_reference_profile(equipment_type)
        if ref_profile:
            assessor.set_normalization_strategy(ReferenceNormalization(ref_profile))
    
    if weight_strategy == "По умолчанию":
        assessor.set_weight_strategy(DefaultWeightStrategy())
    elif weight_strategy == "Для типа оборудования":
        assessor.set_weight_strategy(EquipmentSpecificWeightStrategy())
    else:
        # Custom weights
        st.sidebar.subheader("Настройка весов")
        
        custom_weights = {}
        total_weight = 0
        
        for metric, default_weight in DEFAULT_WEIGHTS.items():
            weight = st.sidebar.slider(
                f"Вес '{metric}'",
                min_value=0.0,
                max_value=1.0,
                value=default_weight,
                step=0.05
            )
            custom_weights[metric] = weight
            total_weight += weight
        
        if abs(total_weight - 1.0) > 0.01:
            st.sidebar.warning(f"Сумма весов должна быть 1.0 (текущая: {total_weight:.2f})")
        else:
            try:
                assessor.set_weight_strategy(CustomWeightStrategy(custom_weights))
            except ValueError as e:
                st.sidebar.error(str(e))
    
    # Calculate technical level
    df_result, error = assessor.calculate_technical_level(data, equipment_type)
    
    if error:
        st.error(f"Ошибка расчета: {error}")
        return
    
    # Display results
    st.subheader("Результаты оценки")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_tech_level = df_result['technical_level'].mean()
        st.metric("Средний тех. уровень", f"{avg_tech_level:.3f}")
    
    with col2:
        max_tech_level = df_result['technical_level'].max()
        st.metric("Максимальный тех. уровень", f"{max_tech_level:.3f}")
    
    with col3:
        min_tech_level = df_result['technical_level'].min()
        st.metric("Минимальный тех. уровень", f"{min_tech_level:.3f}")
    
    # Visualizations
    st.subheader("Визуализация")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown("#### Радиальная диаграмма (сравнение с эталоном)")
        
        # Get reference profile
        ref_profile = assessor.get_reference_profile(equipment_type)
        
        if ref_profile and 'norm_accuracy' in df_result.columns:
            # Prepare data for radar chart
            metrics = ['accuracy', 'price', 'digital_display', 'temperature_range', 'weight']
            
            # Sample equipment (first row or best)
            sample_idx = df_result['technical_level'].idxmax()
            sample = df_result.loc[sample_idx]
            
            categories = []
            sample_values = []
            ref_values = []
            
            for metric in metrics:
                norm_col = f'norm_{metric}'
                if norm_col in df_result.columns:
                    # Russian names for display
                    ru_names = {
                        'accuracy': 'Точность',
                        'price': 'Цена',
                        'digital_display': 'Цифр. дисплей',
                        'temperature_range': 'Темп. диапазон',
                        'weight': 'Вес'
                    }
                    categories.append(ru_names.get(metric, metric))
                    sample_values.append(sample.get(norm_col, 0))
                    ref_values.append(1.0)  # Reference is always 1.0
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=sample_values,
                theta=categories,
                fill='toself',
                name='Образец'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=ref_values,
                theta=categories,
                fill='toself',
                name='Эталон',
                line=dict(dash='dash')
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                template=PLOTLY_TEMPLATE,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Недостаточно данных для радиальной диаграммы")
    
    with viz_col2:
        st.markdown("#### Столбчатая диаграмма (убывание тех. уровня)")
        
        # Sort by technical level descending
        df_sorted = df_result.sort_values('technical_level', ascending=False)
        
        fig = px.bar(
            df_sorted.head(20),
            y='technical_level',
            x=df_sorted.index.astype(str),
            labels={'technical_level': 'Технический уровень', 'index': 'Образец'},
            color='technical_level',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_title="Образец",
            yaxis_title="Технический уровень",
            template=PLOTLY_TEMPLATE,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    if st.checkbox("Показать детальную таблицу"):
        display_cols = [col for col in df_result.columns if col not in ['norm_accuracy', 'norm_price', 'norm_digital_display', 'norm_temperature_range', 'norm_weight']]
        st.dataframe(df_result[display_cols], use_container_width=True)


def render_forecast_tab(data_loader: DataLoader):
    """Render the Forecast tab with regression analysis and predictions."""
    st.header("📈 Прогноз динамики")
    
    data = data_loader.data
    
    if data is None or data.empty:
        st.warning("Сначала загрузите данные на вкладке 'Данные'")
        return
    
    # Initialize forecast engine
    engine = ForecastEngine()
    
    # Correlation analysis
    st.subheader("Корреляционный анализ")
    
    target_col = st.selectbox(
        "Целевая переменная",
        options=['sales', 'price'],
        index=0
    )
    
    corr_matrix = engine.analyze_correlations(data, target_col)
    
    if not corr_matrix.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect='auto',
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(title='Матрица корреляций', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter plots for significant correlations
            significant = engine.get_significant_features(data, target_col)
            
            if significant:
                feature = st.selectbox("Выберите признак для scatter plot", significant)
                
                fig = px.scatter(
                    data,
                    x=feature,
                    y=target_col,
                    trendline='ols',
                    title=f'{feature} vs {target_col}',
                    labels={feature: feature, target_col: target_col}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Нет значимых корреляций (порог > 0.3)")
    
    # Model selection and training
    st.subheader("Обучение модели регрессии")
    
    model_type = st.radio(
        "Тип модели",
        options=["Линейная", "Полиномиальная (степень 2)"],
        index=0
    )
    
    # Select features
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    selected_features = st.multiselect(
        "Признаки для модели",
        options=numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    )
    
    if st.button("Обучить модель"):
        if len(selected_features) < 1:
            st.error("Выберите хотя бы один признак")
        else:
            if "Полиномиальная" in model_type:
                engine.set_strategy(PolynomialRegressionStrategy(degree=2))
            else:
                engine.set_strategy(LinearRegressionStrategy())
            
            metrics, error = engine.fit(data, selected_features, target_col)
            
            if error:
                st.error(f"Ошибка обучения: {error}")
            else:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("R² (train)", f"{metrics['train_r2']:.4f}")
                with col2:
                    st.metric("R² (test)", f"{metrics['test_r2']:.4f}")
                with col3:
                    st.metric("MSE (test)", f"{metrics['test_mse']:.4f}")
                with col4:
                    overfit = "⚠️ Да" if metrics['is_overfitting'] else "✅ Нет"
                    st.metric("Переобучение", overfit)
                
                # Cross-validation
                if st.checkbox("Кросс-валидация"):
                    cv_results = engine.cross_validate(data, selected_features, target_col)
                    
                    if cv_results:
                        st.write(f"**Средний R²:** {cv_results['cv_mean_r2']:.4f} ± {cv_results['cv_std_r2']:.4f}")
                        
                        fig = px.bar(
                            x=['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'][:len(cv_results['cv_scores'])],
                            y=cv_results['cv_scores'][:5],
                            labels={'x': 'Fold', 'y': 'R²'},
                            title='R² по фолдам кросс-валидации'
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction
    st.subheader("Прогнозирование")
    
    pred_col1, pred_col2 = st.columns(2)
    
    with pred_col1:
        st.markdown("#### Точечный прогноз")
        
        input_features = {}
        
        for feature in selected_features:
            if feature in data.columns:
                val = st.number_input(
                    feature,
                    value=float(data[feature].median()),
                    key=f"pred_{feature}"
                )
                input_features[feature] = val
        
        if st.button("Сделать прогноз"):
            if engine.feature_names:
                prediction, error = engine.predict(input_features)
                
                if error:
                    st.error(f"Ошибка прогноза: {error}")
                else:
                    st.success(f"Прогноз {target_col}: **{prediction:.2f}**")
            else:
                st.warning("Сначала обучите модель")
    
    with pred_col2:
        st.markdown("#### Интервальный анализ")
        
        if engine.feature_names:
            var_feature = st.selectbox(
                "Варируемый признак",
                options=selected_features,
                key="var_feature"
            )
            
            # Get fixed values from other features
            fixed_features = input_features.copy()
            
            min_val = st.number_input(
                f"Мин {var_feature}",
                value=float(data[var_feature].min()),
                key=f"min_{var_feature}"
            )
            max_val = st.number_input(
                f"Макс {var_feature}",
                value=float(data[var_feature].max()),
                key=f"max_{var_feature}"
            )
            
            if st.button("Построить график зависимости"):
                result_df, error = engine.interval_analysis(
                    data,
                    fixed_features,
                    var_feature,
                    min_val,
                    max_val
                )
                
                if error or result_df.empty:
                    st.error(f"Ошибка анализа: {error}")
                else:
                    fig = px.line(
                        result_df,
                        x=var_feature,
                        y='predicted_' + target_col,
                        title=f'Зависимость {target_col} от {var_feature}',
                        markers=True
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Сначала обучите модель")


def render_resources_tab(data_loader: DataLoader):
    """Render the Resources tab with optimization."""
    st.header("🔧 Оптимизация ресурсов")
    
    data = data_loader.data
    
    if data is None or data.empty:
        st.warning("Сначала загрузите данные на вкладке 'Данные'")
        return
    
    # Initialize optimizer
    optimizer = ResourceOptimizer()
    
    # Constraints settings
    st.subheader("Параметры оптимизации")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        budget = st.number_input(
            "Бюджет (руб)",
            min_value=0,
            value=int(OPTIMIZATION_DEFAULTS['budget']),
            step=10000
        )
    
    with col2:
        min_demand = st.number_input(
            "Мин. спрос (шт)",
            min_value=0,
            value=int(OPTIMIZATION_DEFAULTS['min_demand']),
            step=1
        )
    
    with col3:
        warehouse_capacity = st.number_input(
            "Ёмкость склада (шт)",
            min_value=1,
            value=int(OPTIMIZATION_DEFAULTS['warehouse_capacity']),
            step=10
        )
    
    if st.button("Выполнить оптимизацию"):
        result_df, error = optimizer.optimize_procurement(
            data,
            budget=budget,
            min_demand=min_demand,
            warehouse_capacity=warehouse_capacity
        )
        
        if error:
            st.error(f"Ошибка оптимизации: {error}")
        elif result_df.empty:
            st.warning("Не удалось найти оптимальное решение")
        else:
            st.subheader("Результаты оптимизации")
            
            summary = optimizer.get_optimization_summary()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Оптимальные затраты", f"{summary.get('total_optimized_cost', 0):,.0f} руб")
            with col2:
                st.metric("Затраты (наивный план)", f"{summary.get('total_naive_cost', 0):,.0f} руб")
            with col3:
                savings = summary.get('savings', 0)
                st.metric("Экономия", f"{savings:,.0f} руб", delta=f"{savings/summary.get('total_naive_cost', 1)*100:.1f}%")
            with col4:
                st.metric("Использование бюджета", f"{summary.get('budget_used', 0):.1f}%")
            
            # Visualization
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.markdown("#### Сравнение плана и оптимума")
                
                # Melt data for comparison
                comparison_df = result_df.melt(
                    id_vars=['equipment_type'],
                    value_vars=['optimal_quantity', 'naive_quantity'],
                    var_name='plan_type',
                    value_name='quantity'
                )
                
                fig = px.bar(
                    comparison_df,
                    x='equipment_type',
                    y='quantity',
                    color='plan_type',
                    barmode='group',
                    labels={'equipment_type': 'Тип оборудования', 'quantity': 'Количество'},
                    color_discrete_map={'optimal_quantity': '#2E86AB', 'naive_quantity': '#A23B72'}
                )
                fig.update_layout(height=400, template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_col2:
                st.markdown("#### Структура затрат")
                
                fig = px.pie(
                    result_df,
                    values='total_cost',
                    names='equipment_type',
                    title='Распределение затрат'
                )
                fig.update_layout(height=400, template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            if st.checkbox("Показать таблицу закупок"):
                st.dataframe(result_df, use_container_width=True)
            
            # Download results
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать результаты (CSV)",
                data=csv,
                file_name='optimization_results.csv',
                mime='text/csv'
            )
    
    # Sensitivity analysis
    st.subheader("Анализ чувствительности")
    
    param = st.selectbox(
        "Параметр для анализа",
        options=['budget', 'min_demand', 'warehouse_capacity'],
        format_func=lambda x: {'budget': 'Бюджет', 'min_demand': 'Мин. спрос', 'warehouse_capacity': 'Ёмкость склада'}[x]
    )
    
    if st.button("Выполнить анализ чувствительности"):
        sensitivity_df, error = optimizer.sensitivity_analysis(data, parameter=param)
        
        if error or sensitivity_df.empty:
            st.error(f"Ошибка анализа: {error}")
        else:
            fig = px.line(
                sensitivity_df,
                x=param,
                y='total_cost',
                title=f'Зависимость затрат от {param}',
                markers=True
            )
            fig.update_layout(height=400, template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Анализ метрологического оборудования",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🏭 Информационно-аналитическая система производителя метрологического оборудования")
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Данные", "🎯 Тех. уровень", "📈 Прогноз", "🔧 Ресурсы"])
    
    with tab1:
        render_data_tab(data_loader)
    
    with tab2:
        render_technical_level_tab(data_loader)
    
    with tab3:
        render_forecast_tab(data_loader)
    
    with tab4:
        render_resources_tab(data_loader)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **Курсовая работа по дисциплине «Программирование на языке Python»**  
        Вариант №29: Производитель метрологического оборудования
        
        Архитектура: MVC + Strategy | Визуализация: Plotly | Оптимизация: SciPy
        """
    )


if __name__ == "__main__":
    main()
