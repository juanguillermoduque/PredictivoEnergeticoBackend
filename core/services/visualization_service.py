import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

import matplotlib
matplotlib.use('Agg')  # Esto debe ir antes de importar pyplot
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import zscore

import base64
from io import BytesIO

class VisualizationService:

    def create_coolwarm(self, df):
        # Seleccionar las columnas correspondientes a las horas del día
        hour_columns = [f'Values_Hour{i:02d}' for i in range(1, 25)]

        # Calcular la matriz de correlación entre las horas
        correlation_matrix = df[hour_columns].corr()
        # Configurar el tamaño de la figura para el heatmap
        plt.figure(figsize=(16, 5))

        # Crear el heatmap con anotaciones y un mapa de colores personalizado
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, linecolor='white')

        # Añadir título y etiquetas a la gráfica
        plt.title('Matriz de Correlación entre las Horas', fontsize=16)
        plt.xlabel('Horas', fontsize=12)
        plt.ylabel('Horas', fontsize=12)

        # Mostrar el gráfico
        plt.show()

        # Guardar el gráfico como imagen
        base_64 = self.guardar_grafico_base64(plt, "matriz_coolwarm", "png")
        return {
            'base_64': base_64
        }
        
    def create_outliers(self, df):
        df['Total_kWh'] = df.select_dtypes(include='number').sum(axis=1)
        df['z_score'] = zscore(df['Total_kWh'])

        df_DemaReal_sistema_final = df

        # Graficar outliers
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_DemaReal_sistema_final['Date'], y=df_DemaReal_sistema_final['Total_kWh'], mode='lines', name='Demanda Real'))
        # linea extra de n

        fig.add_trace(go.Scatter(x=df_DemaReal_sistema_final[df_DemaReal_sistema_final['z_score'].abs() > 2.5]['Date'], y=df_DemaReal_sistema_final[df_DemaReal_sistema_final['z_score'].abs() > 2.5]['Total_kWh'], mode='markers', marker=dict(color='red'), name='Outliers'))
        fig.update_layout(title='Demanda Real [Outliers > 2.5 ZScore]', xaxis_title='Fecha', yaxis_title='Demanda Real (kWh)', showlegend=True)

        # Guardar el gráfico como imagen
        base_64 = self.guardar_grafico_base64(fig, "outler", "png")
        return {
            'base_64': base_64
        }

    def create_demanda_real_vs_z_score(self, df):
        df['Total_kWh'] = df.select_dtypes(include='number').sum(axis=1)
        df['z_score'] = zscore(df['Total_kWh'])

        df_DemaReal_sistema_final = df
        fig = px.scatter(
            df_DemaReal_sistema_final, 
            x='Total_kWh', 
            y='z_score', 
            title='Relación entre Demanda Total y Z-Score',
            labels={'Total_kWh': 'Demanda Total (kWh)', 'z_score': 'Z-Score'},
            template='plotly_white'
        )

        # Personalización del hover con más detalles
        fig.update_traces(
            hovertemplate="<b>Fecha:</b> %{customdata}<br><b>Demanda Total:</b> %{x} kWh<br><b>Z-Score:</b> %{y}<extra></extra>",
            customdata=df_DemaReal_sistema_final['Date']  # Fecha del punto
        )

        # Mejorar el diseño del gráfico
        fig.update_layout(
            title={'text': 'Relación entre Demanda Total y Z-Score', 'x': 0.5},
            xaxis_title='Demanda Total (kWh)',
            yaxis_title='Z-Score',
            font=dict(size=14),
            showlegend=False
        )
        
        # Guardar el gráfico como imagen
        base_64 = self.guardar_grafico_base64(fig, "demanda_real_vs_z_score", "png")
        return {
            'base_64': base_64
        }

    def create_box_plot(self, df):
        df['Total_kWh'] = df.select_dtypes(include='number').sum(axis=1)
        df['z_score'] = zscore(df['Total_kWh'])
        df_DemaReal_sistema_final = df
        fig_box = px.box(
            df_DemaReal_sistema_final, 
            y='Total_kWh', 
            title='Box Plot de Demanda Real del Sistema (kWh)',
            points='all',  # Mostrar todos los puntos
            notched=True,  # Añadir muescas para visualizar la mediana
            color_discrete_sequence=['#636EFA']  # Color personalizado
        )

        # Personalizar el diseño del gráfico
        fig_box.update_layout(
            title_text='Box Plot de Demanda Real del Sistema (kWh)',
            title_x=0.5,
            yaxis_title_text='Total_kWh',
            template='plotly_white',
            font=dict(size=14),
            showlegend=False
        )

        # Personalización del hover con más detalles
        fig_box.update_traces(
            hovertemplate="<b>Fecha:</b> %{customdata}<br><b>Cantidad de Energía:</b> %{y} kWh<extra></extra>",
            customdata=df_DemaReal_sistema_final['Date']  # Fecha del punto
        )

        # Guardar el gráfico como imagen
        base_64 = self.guardar_grafico_base64(fig_box, "box_plot", "png")
        return {
            'base_64': base_64
        }

    def create_histograma(self, df):
        df['Total_kWh'] = df.select_dtypes(include='number').sum(axis=1)
        df['z_score'] = zscore(df['Total_kWh'])
        df_DemaReal_sistema_final = df
        fig_hist = px.histogram(
            df_DemaReal_sistema_final, 
            x='Total_kWh', 
            nbins=50, 
            title='Histograma de Demanda Real del Sistema (kWh)',
            color_discrete_sequence=['#636EFA']
        )

        # Personalizar el diseño del histograma
        fig_hist.update_layout(
            title_text='Histograma de Demanda Real del Sistema (kWh)',
            title_x=0.5,
            xaxis_title_text='Total_kWh',
            yaxis_title_text='Frecuencia',
            template='plotly_white',
            font=dict(size=14),
            bargap=0.1
        )

        # Añadir líneas de referencia para la media y la mediana
        mean_value = df_DemaReal_sistema_final['Total_kWh'].mean()
        median_value = df_DemaReal_sistema_final['Total_kWh'].median()

        fig_hist.add_vline(x=mean_value, line_width=3, line_dash="dash", line_color="green", annotation_text="Media", annotation_position="top left")
        fig_hist.add_vline(x=median_value, line_width=3, line_dash="dash", line_color="red", annotation_text="Mediana", annotation_position="top right")

         # Guardar el gráfico como imagen
        base_64 = self.guardar_grafico_base64(fig_hist, "histograma", "png")
        return {
            'base_64': base_64
        }

    def create_warm_map(self, df):
        df['Total_kWh'] = df.select_dtypes(include='number').sum(axis=1)
        df['z_score'] = zscore(df['Total_kWh'])
        df_DemaReal_sistema_final = df

        # Filtrar las columnas necesarias y establecer 'Date' como índice
        df_heatmap = df_DemaReal_sistema_final[['Date'] + [f'Values_Hour{i:02d}' for i in range(1, 25)]].set_index('Date')

        # Crear el heatmap utilizando plotly
        fig = px.imshow(df_heatmap.T, 
                        labels=dict(x="Fecha", y="Hora del Día", color="Demanda (kWh)"),
                        title='Distribución Horaria de la Demanda',
                        aspect="auto")

        # Personalizar el diseño del heatmap
        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Hora del Día",
            coloraxis_colorbar=dict(title="Demanda (kWh)"),
            template='plotly_white',
            width=1100,  # Ancho de la figura
            height=550   # Altura de la figura
        )
        #Se guarda el gráfico como imagen
        base_64 = self.guardar_grafico_base64(fig, "warm_map", "png")
        return {
            'base_64': base_64
        }
    
    def create_acumilativo_demanda_horaria(self, df):
        df['Total_kWh'] = df.select_dtypes(include='number').sum(axis=1)
        df['z_score'] = zscore(df['Total_kWh'])
        df_DemaReal_sistema_final = df
        fig = px.area(
            df_DemaReal_sistema_final, 
            x='Date', 
            y=[f'Values_Hour{i:02d}' for i in range(1, 25)], 
            title='Demanda Horaria a lo Largo del Día'
        )

        # Ajustar el diseño del gráfico
        fig.update_layout(
            width=1100, 
            height=550, 
            title_x=0.5,  # Centrar el título
            xaxis_title='Fecha',
            yaxis_title='Demanda (kWh)',
            template='plotly_white',  # Usar un tema claro
            font=dict(size=14)  # Ajustar el tamaño de la fuente
        )

        #guardar el gráfico como imagen
        base_64 = self.guardar_grafico_base64(fig, "acumilativo_demanda_horaria", "png")
        return {
            'base_64': base_64
        }

    def create_line_demanda_promedio(self, df):
        df['Total_kWh'] = df.select_dtypes(include='number').sum(axis=1)
        df['z_score'] = zscore(df['Total_kWh'])
        df_DemaReal_sistema_final = df
        # Calcular el promedio de cada hora
        hourly_means = df_DemaReal_sistema_final[[f'Values_Hour{i:02d}' for i in range(1, 25)]].mean()

        # Crear un DataFrame para los promedios horarios
        hourly_means_df = pd.DataFrame({
            'Hour': range(1, 25),
            'Average_Demand': hourly_means.values
        })

        # Graficar el comportamiento promedio de cada hora
        fig = px.line(hourly_means_df, x='Hour', y='Average_Demand', title='Comportamiento Promedio de Cada Hora')
        fig.update_layout(
            xaxis_title='Hora del Día', 
            yaxis_title='Demanda Promedio (kWh)',
            width=1100,  # Ancho del gráfico
            height=500   # Altura del gráfico
        )

        # Guardar el gráfico como imagen
        base_64 = self.guardar_grafico_base64(fig, "line_demanda_promedio", "png")
        return {
            'base_64': base_64
        }
    
    def create_barras_demanda_promedio(self, df):
        df['Total_kWh'] = df.select_dtypes(include='number').sum(axis=1)
        df['z_score'] = zscore(df['Total_kWh'])
        df_DemaReal_sistema_final = df
        # Calcular el promedio de cada hora
        avg_hourly = df_DemaReal_sistema_final[[f'Values_Hour{i:02d}' for i in range(1, 25)]].mean()

        # Crear un DataFrame para los promedios horarios
        avg_hourly_df = pd.DataFrame({
            'Hour': range(1, 25),
            'Average_Demand': avg_hourly.values
        })

        # Crear el gráfico de barras
        fig = px.bar(
            avg_hourly_df,
            x='Hour',
            y='Average_Demand',
            title='Demanda Promedio por Hora',
            labels={'Hour': 'Hora del Día', 'Average_Demand': 'Demanda Promedio (kWh)'},
            template='plotly_white'
        )

        # Personalizar el diseño del gráfico
        fig.update_layout(
            width=1100,  # Ancho del gráfico
            height=500,  # Altura del gráfico
            xaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=1
            ),
            yaxis_title='Demanda Promedio (kWh)',
            xaxis_title='Hora del Día'
        )

        #guardar el gráfico como imagen
        base_64 = self.guardar_grafico_base64(fig, "barras_demanda_promedio", "png")
        return {
            'base_64': base_64
        }
    
    def create_barras_demanda_promedio_mes(self, df):
        df['Total_kWh'] = df.select_dtypes(include='number').sum(axis=1)
        df['z_score'] = zscore(df['Total_kWh'])
        df_DemaReal_sistema_final = df
        df_DemaReal_sistema_final['Month'] = df_DemaReal_sistema_final['Date'].dt.month
        df_monthly = df_DemaReal_sistema_final.groupby('Month')['Total_kWh'].mean().reset_index()
        fig = px.bar(df_monthly, x='Month', y='Total_kWh', title='Promedio Diario por Mes de Demanda Total')

        #guardar el gráfico como imagen
        base_64 = self.guardar_grafico_base64(fig, "barras_demanda_promedio_mes", "png")
        return {
            'base_64': base_64,
        }
    
    def create_barras_demanda_promedio_dia(self, df):
        df['Total_kWh'] = df.select_dtypes(include='number').sum(axis=1)
        df['z_score'] = zscore(df['Total_kWh'])
        df_DemaReal_sistema_final = df

        # Añadir columna de semana del año
        df_DemaReal_sistema_final['Week'] = df_DemaReal_sistema_final['Date'].dt.isocalendar().week

        # Agrupar por semana y calcular el promedio de 'Total_kWh'
        df_weekly = df_DemaReal_sistema_final.groupby('Week')['Total_kWh'].mean().reset_index()

        # Crear la gráfica de barras
        fig = px.bar(df_weekly, x='Week', y='Total_kWh', title='Promedio Diario por Semana del Año de Demanda Total')

        # Guardar el gráfico como imagen
        base_64 = self.guardar_grafico_base64(fig, "barras_demanda_promedio_dia", "png")
        return {
            'base_64': base_64,
        }
    
    def create_barras_demanda_promedio_anio(self, df):
        df['Total_kWh'] = df.select_dtypes(include='number').sum(axis=1)
        df['z_score'] = zscore(df['Total_kWh'])
        df_DemaReal_sistema_final = df
        # Añadir columna de año
        df_DemaReal_sistema_final['Year'] = df_DemaReal_sistema_final['Date'].dt.year

        # Agrupar por año y calcular el promedio de 'Total_kWh'
        df_yearly = df_DemaReal_sistema_final.groupby('Year')['Total_kWh'].mean().reset_index()

        # Crear la gráfica de barras
        fig = px.bar(df_yearly, x='Year', y='Total_kWh', title='Promedio Diario por Año de Demanda Total')

        # Guardar el gráfico como imagen
        base_64 = self.guardar_grafico_base64(fig, "barras_demanda_promedio_anio", "png")
        return {
            'base_64': base_64,
        }
    
    def create_barras_acumulado_promedio_anio_mes(self, df):
        df['Total_kWh'] = df.select_dtypes(include='number').sum(axis=1)
        df['z_score'] = zscore(df['Total_kWh'])
        df_DemaReal_sistema_final = df
        # Asegúrate de que la columna 'Date' esté en formato datetime
        df_DemaReal_sistema_final['Date'] = pd.to_datetime(df_DemaReal_sistema_final['Date'])

        # Crear columnas de año y mes
        df_DemaReal_sistema_final['Year'] = df_DemaReal_sistema_final['Date'].dt.year
        df_DemaReal_sistema_final['Month'] = df_DemaReal_sistema_final['Date'].dt.month

        # Crear una tabla dinámica donde cada mes sea una columna
        df_monthly = df_DemaReal_sistema_final.pivot_table(
            index='Year', 
            columns='Month', 
            values='Total_kWh', 
            aggfunc='sum', 
            fill_value=0
        ).reset_index()

        # Renombrar las columnas para mayor claridad
        df_monthly.columns = ['Año'] + [f'Mes_{i}' for i in range(1, 13)]

        # Crear la gráfica de barras
        fig = px.bar(df_monthly, 
                    x='Año', 
                    y=[f'Mes_{i}' for i in range(1, 13)], 
                    title='Consumo Total por Mes y Año',
                    labels={'Año': 'Año', 'value': 'Consumo Total (kWh)', 'variable': 'Mes'},
                    barmode='group')

        # Modificar las dimensiones del plot
        fig.update_layout(
            width=1100,  # Ancho del plot
            height=450   # Altura del plot
        )

        # Guardar el gráfico como imagen
        base_64 = self.guardar_grafico_base64(fig, "barras_acumulado_promedio_anio_mes", "png")
        return {
            'base_64': base_64,
        }

    def create_dispersion_horaria(self, df):
        df['Total_kWh'] = df.select_dtypes(include='number').sum(axis=1)
        df['z_score'] = zscore(df['Total_kWh'])
        df_DemaReal_sistema_final = df

        fig = px.scatter(
            df_DemaReal_sistema_final, 
            x='Values_Hour05', 
            y='Values_Hour17', 
            title='Demanda 5 AM vs Demanda 5 PM',
            labels={'Values_Hour05': 'Demanda a las 5 AM', 'Values_Hour17': 'Demanda a las 5 PM'},
            template='plotly_white'
        )

        # Personalización del hover con más detalles
        fig.update_traces(
            hovertemplate="<b>Fecha:</b> %{customdata}<br><b>Demanda a las 5 AM:</b> %{x} kWh<br><b>Demanda a las 5 PM:</b> %{y} kWh<extra></extra>",
            customdata=df_DemaReal_sistema_final['Date']  # Fecha del punto
        )

        # Mejorar el diseño del gráfico
        fig.update_layout(
            title={'text': 'Demanda 5 AM vs Demanda 5 PM', 'x': 0.5},
            xaxis_title='Demanda a las 5 AM (kWh)',
            yaxis_title='Demanda a las 5 PM (kWh)',
            font=dict(size=14),
            showlegend=False
        )
        # Guardar el gráfico como imagen
        base_64 = self.guardar_grafico_base64(fig, "dispersion_horaria", "png")
        return {
            'base_64': base_64
        }
    
    def create_historico_minimo_maximo(self, df):
        df['Total_kWh'] = df.select_dtypes(include='number').sum(axis=1)
        df['z_score'] = zscore(df['Total_kWh'])
        df_DemaReal_sistema_final = df

        # Función para configurar el layout de la gráfica


        # 1. Comprobación de valores nulos
        if df_DemaReal_sistema_final.isnull().any().any():
            print("Advertencia: El DataFrame contiene valores nulos. Se procederá a eliminar los registros.")
            df_DemaReal_sistema_final = df_DemaReal_sistema_final.dropna()  # Eliminar los registros con valores nulos
            # Alternativamente, podrías reemplazar los valores nulos si lo deseas

        # 3. Crear la gráfica
        fig = go.Figure()  # Crear una nueva figura para la gráfica

        # 4. Añadir los datos al gráfico
        fig.add_trace(go.Scatter(
            x=df_DemaReal_sistema_final['Date'],  # Fecha en el eje X
            y=df_DemaReal_sistema_final['Total_kWh'],  # Consumo de energía en el eje Y
            mode='lines',  # Tipo de gráfico: líneas
            name='Demanda Real'  # Nombre de la serie en la leyenda
        ))

        # Encontrar el punto mínimo y máximo del consumo de energía
        min_point = df_DemaReal_sistema_final.loc[df_DemaReal_sistema_final['Total_kWh'].idxmin()]  # Punto mínimo
        max_point = df_DemaReal_sistema_final.loc[df_DemaReal_sistema_final['Total_kWh'].idxmax()]  # Punto máximo

        # Personalización del hover con más detalles
        fig.update_traces(
            hovertemplate="<b>Fecha:</b> %{x}<br><b>Cantidad de Energía:</b> %{y} kWh<br><b>Media Móvil (7 días):</b> %{customdata}<extra></extra>",  # Detalles que aparecerán al pasar el ratón
            customdata=df_DemaReal_sistema_final['Total_kWh'].rolling(window=7).mean()  # Media móvil de 7 días para el hover
        )

        # Añadir el punto mínimo en verde
        fig.add_trace(go.Scatter(
            x=[min_point['Date']],  # Fecha del punto mínimo
            y=[min_point['Total_kWh']],  # Consumo en el punto mínimo
            mode='markers',  # Mostrarlo como un marcador
            marker=dict(color='green', size=8),  # Personalizar el marcador (color verde, tamaño 8)
            name='Mínimo',  # Nombre del punto en la leyenda
            opacity=0.5,  # Transparencia del marcador
            hoverinfo='skip'  # Evitar que el punto sea clickeable
        ))

        # Añadir el punto máximo en rojo
        fig.add_trace(go.Scatter(
            x=[max_point['Date']],  # Fecha del punto máximo
            y=[max_point['Total_kWh']],  # Consumo en el punto máximo
            mode='markers',  # Mostrarlo como un marcador
            marker=dict(color='red', size=8),  # Personalizar el marcador (color rojo, tamaño 8)
            name='Máximo',  # Nombre del punto en la leyenda
            opacity=0.5,  # Transparencia del marcador
            hoverinfo='skip'  # Evitar que el punto sea clickeable
        ))

        # 5. Configurar el layout de la gráfica
        self.configurar_layout(fig, 'Demanda Real del Sistema 2022-2024', 'Fecha', 'Cantidad de Energía (kWh)')

        # 6. Establecer el rango de fechas dinámico en el slider
        min_date = df_DemaReal_sistema_final['Date'].min()  # Fecha mínima
        max_date = df_DemaReal_sistema_final['Date'].max()  # Fecha máxima

        # Configuración del eje X con un slider para ajustar el rango de fechas
        fig.update_xaxes(
            tickformat="%d-%m-%Y",  # Formato de las fechas
            rangeslider_visible=True,  # Mostrar el slider para seleccionar un rango de fechas
            rangeselector=dict(
                buttons=list([  # Botones de selección rápida
                    dict(count=7, label="1S", step="day", stepmode="backward"),  # Seleccionar 1 semana
                    dict(count=1, label="1M", step="month", stepmode="backward"),  # Seleccionar 1 mes
                    dict(count=3, label="3M", step="month", stepmode="backward"),  # Seleccionar 3 meses
                    dict(count=6, label="6M", step="month", stepmode="backward"),  # Seleccionar 6 mesesq
                    dict(count=1, label="1A", step="year", stepmode="backward"),  # Seleccionar 1 año
                    dict(count=2, label="2A", step="year", stepmode="backward"),  # Seleccionar 2 años
                    dict(step="all", label="Todo")  # Seleccionar todo el rango
                ]),
                x=1.5,  # Posición de los botones en el eje X (centro)
                xanchor="center",  # Anclaje de los botones en el centro
                y=1.5,  # Posición de los botones en el eje Y (debajo de la gráfica)
                yanchor="top",  # Anclaje de los botones en la parte superior
                font=dict(size=20)  # Tamaño de la fuente de los botones
            ),
            range=[min_date, max_date]  # Rango de fechas inicial
        )

        # 7. Añadir botones para mostrar/ocultar la línea de la demanda
        fig.update_layout(
            updatemenus=[  # Lista de menús desplegables (en este caso, botones)
                dict(
                    type="buttons",  # Tipo de menú: botones
                    direction="right",  # Dirección de los botones
                    buttons=list([  # Lista de botones
                        dict(
                            args=[{"visible": [False, True, True, True]}],  # Ocultar la línea de la métrica real
                            label="Ocultar métrica real",  # Etiqueta para el botón
                            method="update"  # Acción de actualización cuando se hace clic
                        ),
                        dict(
                            args=[{"visible": [True, True, True, True]}],  # Mostrar la línea de la métrica real
                            label="Mostrar métrica real",  # Etiqueta para el botón
                            method="update"  # Acción de actualización cuando se hace clic
                        )
                    ]),
                    pad={"r": 6, "t": 6},  # Espaciado alrededor de los botones
                    showactive=True,  # Mantener el botón activo cuando se selecciona
                    x=1.5,  # Posición de los botones en el eje X (centro)
                    xanchor="center",  # Anclaje de los botones en el centro
                    y=2.5,  # Posición de los botones en el eje Y (por encima de la gráfica)
                    yanchor="top",  # Anclaje de los botones en la parte superior
                    font=dict(size=16),  # Tamaño de la fuente de los botones
                    # title=dict(text="Opciones de visualización", font=dict(size=16))  # Título del conjunto de botones
                ),
            ]
        )
        #guardar el gráfico como imagen
        base_64 = self.guardar_grafico_base64(fig, "historico_minimo_maximo", "png")
        return {
            'base_64': base_64
        }

    def configurar_layout(self,fig, title, x_title, y_title):
        """
        Configura el diseño de la gráfica, incluyendo títulos, etiquetas de los ejes, fuente y leyenda.
        """
        fig.update_layout(
            title=title,  # Título de la gráfica
            xaxis_title=x_title,  # Título del eje X
            yaxis_title=y_title,  # Título del eje Y
            template='plotly_white',  # Plantilla de color blanco para la gráfica
            font=dict(size=12),  # Configuración del tamaño de la fuente
            legend=dict(
                title="Elementos",  # Título de la leyenda
                orientation="h",  # Orientación horizontal de la leyenda
                yanchor="top",  # Anclaje de la leyenda en la parte superior
                y=2,  # Posición de la leyenda en el eje Y
                xanchor="left",  # Anclaje de la leyenda a la izquierda
                x=1  # Posición de la leyenda en el eje X
            ),
            width=1150,  # Ajustar el tamaño de la figura (ancho)
            height=500  # Ajustar el tamaño de la figura (alto)
        )
        
    def guardar_grafico_base64(self, fig, filename="grafico", format="png"):
        """
        Guarda la figura (matplotlib o plotly) y la retorna como base64.
        """
        try:
            buffer = BytesIO()
            # Detectar tipo de figura
            if hasattr(fig, 'savefig'):  # matplotlib
                fig.savefig(buffer, format=format)
            elif hasattr(fig, 'write_image'):  # plotly
                fig.write_image(buffer, format=format)
            else:
                raise ValueError("Tipo de figura no soportado")

            buffer.seek(0)
            base_64 = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()
            return base_64

        except Exception as e:
            print(f"Error al guardar el gráfico: {e}")
            return None