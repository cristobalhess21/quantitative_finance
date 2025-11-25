import pandas as pd
import numpy as np
import scipy.stats as stats 
from .backend import market_prices  

def porfolio_volatility(
        df: pd.DataFrame,
        vector_w:np.array
        ) -> float:
    '''
    Calculo de la volatilidad de un portafolio
    de inversiones

    df (pd.DataFrame)
        Dataframe de Retornos del portafolio.
    vector_w (np.array)
        Vector de pesos del los instrumentos del portafolio.
    
    Return (float): volatilidad del portafolio.
    ''' 

    # Matriz varianza - covarianza
    m_cov = df.cov()

    # Vector transpuesto
    vector_w_t = np.array([vector_w])

    # Varianza 
    vector_cov = np.dot(m_cov, vector_w)
    varianza = np.dot(vector_w_t, vector_cov)

    # Volatilidad
    vol = np.sqrt(varianza)

    return vol[0]

def portafolio_returns(
        tickers: list,
        start: str,
        end: str,
        ) -> pd.DataFrame: 
    '''
    Descarga desde la base de datos los precios
    de los intrumentos indicados en en el rango de fechas.

    tickers (list): 
        Lista de nemos de instrumentos que componen
        el portafolio.

    start (str):
        Fecha de inicio de precios

    end (str):
        Fecha de término de precios 

    Return (pd.DataFrame): 
        Dataframe de retornos diarios. 
    '''

    # Descarga de precios
    df = market_prices(
        start_date=start, 
        end_date=end, 
        tickers=tickers
        )

    # Pivot retornos 
    df_pivot = pd.pivot_table(
        data=df, 
        index='FECHA', 
        columns='TICKER', 
        values='PRECIO_CIERRE',
        aggfunc='max'
    )
    df_pivot = df_pivot.pct_change().dropna()   

    return df_pivot

def VaR(sigma:float, confidence:float) -> float:
    '''
    Calculo del Value at Risk al nivel de 
    confianza indicado. Con supuesto de media cero
    '''

    # Estadistico z al nivel de confianza
    z_score = stats.norm.ppf(confidence)

    # VaR
    var = z_score * sigma     

    return var  