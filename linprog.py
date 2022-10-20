import numpy as np
from texttable import Texttable

def simplex_l( c, A, b, B_l, return_base = False ):

  m, n = A.shape

  N_l = [ i for i in range( n ) if i not in B_l ]

  B = A[ :, B_l ]
  x_B = np.linalg.solve( B, b )

  retval = {}

  while True:

    N = A[ :, N_l ]

    c_N = c[ N_l ]
    c_B = c[ B_l ]

    v = np.linalg.solve( B.T, c_B )
    z = c_N - v @ N

    i = np.argmin( z )
    if z[ i ] >= 0.0:
      if return_base:
        return B_l
      x = np.zeros( ( n, ) )
      x[ B_l ] = x_B
      retval[ 'status' ] = 'Sucesso'
      retval[ 'x' ] = x
      return retval

    y = np.linalg.solve( B, N[ :, i ] )

    if y[ np.argmax( y ) ] <= 0:
      if return_base:
        return B_l
      x = np.zeros( ( n, ) )
      x[ B_l ] = x_B
      d = np.zeros( ( n, ) )
      d[ B_l ] = y
      d[ N_l[ i ] ] = 1.0
      retval[ 'status' ] = 'Ilimitado'
      retval[ 'x' ] = x
      retval[ 'd' ] = d
      return retval
    
    eps = float( 'inf' )
    l = None
    for k in range( m ):
      if y[ k ] > 0:
        if x_B[ k ] / y[ k ] < eps:
          eps = x_B[ k ] / y[ k ]
          l = k
    
    x_B = x_B - y * eps
    x_B[ l ] = eps
    tmp = B_l[ l ]
    B_l[ l ] = N_l[ i ]

    N_l[ i ] = tmp
    B = A[ :, B_l ]

def simplex( c, A, b ):

  m, n = A.shape
  c_tilde = np.zeros( ( n + m, ) )
  c_tilde[ n :  ] = 1.0

  I_tilde = np.zeros( ( m, m ) )
  I_tilde[ range( m ), range( m ) ] = b

  A_tilde = np.empty( ( m, n + m ) )
  A_tilde[ :, : n ] = A
  A_tilde[ :, n : ] = I_tilde

  B_l = list( range( n, n + m ) )

  B_l = simplex_l( c_tilde, A_tilde, b, B_l, return_base = True )

  retval = {}
  if max( B_l ) >= n:
    retval[ 'status' ] = 'Invabilidade detectada'
    return retval

  return simplex_l( c, A, b, B_l )

c = np.array([
  -900, -850, -250, -650, -50, -50, -250, -180, -80, -200, -300, -600, -1000, -400, -350, -450,
  -550, -80, -150, -750, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
])

A_ub = np.array([
  [24, 25, 12, 20, 4, 3, 4, 3, 2, 3, 6, 8, 10, 30, 10, 10, 10, 5, 8,25,                                      ],
  [4, 5, 1, 3, 0.25, 0.25, 0.15, 0.2, 0.1, 3, 3, 2, 1, 3, 3, 3, 0.6, 0.2, 3, 2,                              ],
  [8, 8, 0, 0.5, 0.5, 0, 2, 1, 0.5, 0, 1, 3, 0, 0, 0, 1, 0, 0.35, 3.5,2,                                     ],
  [0.2, 0.2, 0.1, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.2, 0.1, 0.2, 0.2, 0.1, 0, 0,0.1, 0.1,                          ],
  [0.1, 0.1, 0.05, 0.1, 0.01, 0, 0.5, 0.01, 0.01, 0, 0.01, 0.02, 0.01, 0.05,0.1, 0.1, 0.7, 0.01, 0.02, 0.1,  ],
  [0.5, 0, 0.5, 0, 0, 0, 0, 0, 0.4, 3.5, 3.5, 0, 1, 0.2, 0.7, 0.7, 0, 0.1, 0, 0,                             ],
  [0, 0, 0, 4, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0,                                       ],
  [0, 0, 20, 0, 0, 6, 0, 0, 0, 5, 0, 0, 2, 5, 3, 0.1, 18, 2, 0, 20,                                          ],  
  [4, 4, 10, 15, 3, 3, 2, 2, 1.5, 7, 12, 12, 30, 20, 25, 20, 1, 7, 7, 30,                                    ], 
  [0, 0, 1, 0, 0, 0.5, 0, 0, 0, 1, 0, 0, 0.2, 0.5, 0.4, 0.2, 0.4, 0.1, 2, 3,                                 ],
  [0, 0, 0, 0, 0, 0, 50, 0, 100, 0, 100, 100, 0, 100, 0, 0.1, 0, 50, 150, 0,                                 ],
])

# Recursos disponíveis
b_ub = np.array([
  70000,  # homens/hora/mes
  9000,  # capacidade de estoque
  40000,  # madeira MDF
  50000,  # plástico
  70000,  # parafusos / pregos / rebites (kg)
  30000,  # vidro temperado m²
  40000,  # tecido (m²)
  60000,  # aço (kg)
  100000,  # horas de máquina
  30000,  # Tinta (L)
  30000,  # cola (g)
  -40,
  -40,
  -9,
  -5,
  -40,
  -40,
  -20,
  -10,
  -30,
  -10,
  -10,
  -10,
  -30,
  -10,
  -20,
  -10,
  -9,
  -40,
  -30,
  -7
])

# Restrição de horas de produção para cada equipamento:
# 	Quantidade minima de equipamentos que devo produzir
ArmarioMDF       = (40, None)
GuardaRoupas     = (40, None)
Fogao            = (9 , None)
Sofa             = (5 , None)
CadeiraMadeira   = (40, None)
CadeiraAco       = (40, None)
PainelTV         = (20, None)
Escrivaninha     = (10, None)
Gabinete         = (30, None)
MesaAco          = (10, None)
MesaMadeira      = (10, None)
CamaMadeira      = (10, None)
MesaVidro        = (30, None)
ArmarioAco       = (10, None)
JanelaAco        = (20, None)
JanelaMadeira    = (10, None)
EstanteAco       = (9 , None)
TabuaPassarRoupa = (40, None)
Sapateira        = (30, None)
CamaAco          = (7 , None)

bounds = [
  ArmarioMDF, GuardaRoupas, Fogao, Sofa, CadeiraMadeira, CadeiraAco, PainelTV,
  Escrivaninha, Gabinete, MesaAco, MesaMadeira, CamaMadeira, MesaVidro,
  ArmarioAco, JanelaAco, JanelaMadeira, EstanteAco, TabuaPassarRoupa,
  Sapateira, CamaAco
]

equipamento = [
  "Armário MDF", "Guarda-Roupas", "Fogão", "Sofá", "Cadeira de Madeira",
  "Cadeira Aço", "Painel de TV", "Escrivaninha", "Gabinete", "Mesa de Aço",
  "Mesa de Madeira", "Cama de Madeira", "Mesa de Vidro", "Armário de Aço",
  "Janela de Aço", "Janela de Madeira", "Estante de Aço",
  "Tábua e Passar Roupa", "Sapateira", "Cama de Aço"
]

itens = [
  "horas/homens",
  "unidades m²/estoque",
  "madeira MDF (m²)",
  "plastico (kg)",
  "parafusos / pregos / rebites (kg)",
  "vidro temperado m²",
  "tecido (m²)",
  "aço (kg)",
  "horas de máquina",
  "Tinta (L)",
  "cola (g)",
]

m, n = A_ub.shape

bounds_tilde = np.zeros((n,n))
np.fill_diagonal(bounds_tilde, -1)

slack_variables = np.zeros((m+n, m+n))
np.fill_diagonal(slack_variables, 1)

A_ub = np.concatenate((A_ub,bounds_tilde),axis=0)

A_ub = np.concatenate((A_ub,slack_variables),axis=1)

# print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
#       for row in A_ub]))

resultado = simplex(c, A_ub, b_ub)

print(resultado['status'])

x = resultado['x']

sobra_itens_table = Texttable()
sobra_itens_table.add_row(["item", "quantidade"])

index = 0;

for value in x[:n]:
  sobra_itens_table.add_row([equipamento[index], "{:.0f}".format(value)])
  index += 1

sobra_itens_table.set_cols_align(['r', 'l'])
print(sobra_itens_table.draw())

