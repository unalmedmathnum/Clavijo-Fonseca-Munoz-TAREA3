import math
import pandas as pd
import matplotlib.pyplot as plt

# funcion que evalua el sistema de ecuaciones con los siguientes parametros: ε = 0.2, λ = 0.1, Ω=0.5 (Parametros similares a los de la figura 4b del articulo)
def F(t, u1, u2):
  LAMBDA = 0.1
  EPSILON = 0.2
  OMEGA = 0.5
  return (u2,-(1+EPSILON*LAMBDA*OMEGA**2*math.cos(OMEGA*t))*(u1-(EPSILON**2*u1**3)/6))

def rungeKutta4(w0, v0, a, b, n: int, f = F):
    """
    Aproxima la solución de la EDO: w' = F(t, W)

    Parámetros:
      - f: función que define la EDO, F(t, W). Esta funcion debe retornar una tupla con los valores de evaluar en el primer y segundo sistema de ecuaciones (w,v)
      - w0: primera condicion inicial
      - v0: segunda condicion inicial
      - a: tiempo inicial
      - b: tiempo final
      - n: número de pasos

    Retorna:
      - lista de aproximaciones para w en cada paso.
    """
    h = (b - a) / n
    t = [a]  # Lista con los tiempos t_i
    w = [w0]  # Lista con las aproximaciones w_i
    v = [v0]  # Lista con las aproximaciones v_i

    for i in range(0, n):
        t_i = a + h * i

        # Calculo de los k_i para la primera y segunda ecuacion del sistema de ecuaciones:
        kw_1 = h * f(t_i, w[i], v[i])[0]
        kv_1 = h * f(t_i, w[i], v[i])[1]
        kw_2 = h * f(t_i + h / 2, w[i] + kw_1 / 2, v[i] + kv_1 / 2)[0]
        kv_2 = h * f(t_i + h / 2, w[i] + kw_1 / 2, v[i] + kv_1 / 2)[1]
        kw_3 = h * f(t_i + h / 2, w[i] + kw_2 / 2, v[i] + kv_2 / 2)[0]
        kv_3 = h * f(t_i + h / 2, w[i] + kw_2 / 2, v[i] + kv_2 / 2)[1]
        kv_4 = h * f(t_i + h, w[i] + kw_3, v[i] + kv_3)[1]
        kw_4 = h * f(t_i + h, w[i] + kw_3, v[i] + kv_3)[0]

        # Agregar w_(i+1) y v_(i+1) a la lista de aproximaciones:
        t.append(t_i + h)
        w.append(w[i] + (kw_1 + 2 * kw_2 + 2 * kw_3 + kw_4) / 6)
        v.append(v[i] + (kv_1 + 2 * kv_2 + 2 * kv_3 + kv_4) / 6)

    return t, w, v

if __name__ == '__main__':
    t, w, v = rungeKutta4(0.5, 0, 0, 100, 10000)
    df = pd.DataFrame({'τ':t, 'u_1':w}).set_index('τ')
    print(df)
    plt.plot(df.index,df, label = '$u_1$', color = 'red') # Ingresar un dataframe df con los valores a graficar, en el indice del dataframe estan los t_i
    plt.legend(loc='upper left') # para hacer que el label salga en la esquina superior izquierda
    plt.title('Solucion con metodo de runge-kutta de orden 4')
    plt.xlabel('τ')
    plt.ylabel('u')
    plt.grid(True)
    plt.show()