# Dicionário de Dados - flights.csv

Este documento detalha as variáveis presentes no conjunto de dados de atrasos e cancelamentos de voos.

## Estrutura dos Dados

| Coluna | Descrição | Tipo / Unidade |
| :--- | :--- | :--- |
| YEAR | Ano do voo (ex.: 2015) [cite: 3] | Inteiro |
| MONTH | Mês do voo (1 a 12) [cite: 3] | Inteiro |
| DAY | Dia do mês do voo (1 a 31) [cite: 3] | Inteiro |
| DAY_OF_WEEK | Dia da semana (1 = Segunda, 7 = Domingo) [cite: 3] | Inteiro |
| AIRLINE | Código da companhia aérea (ex.: AA = American Airlines) [cite: 3] | Categórica |
| FLIGHT_NUMBER | Número do voo [cite: 3] | Inteiro |
| TAIL_NUMBER | Número de registro da aeronave [cite: 3] | Texto |
| ORIGIN_AIRPORT | Código IATA do aeroporto de origem (ex.: ATL) [cite: 3] | Categórica |
| DESTINATION_AIRPORT | Código IATA do aeroporto de destino [cite: 3] | Categórica |
| SCHEDULED_DEPARTURE | Horário de partida programado (HHMM) [cite: 3] | Inteiro |
| DEPARTURE_TIME | Horário real de partida (HHMM) [cite: 3] | Inteiro |
| DEPARTURE_DELAY | Atraso na partida (em minutos) [cite: 3] | Numérico |
| TAXI_OUT | Tempo gasto taxiando até a decolagem (em minutos) [cite: 3] | Numérico |
| WHEELS_OFF | Horário em que o avião decolou (HHMM) [cite: 3] | Inteiro |
| SCHEDULED_TIME | Tempo total programado de voo (em minutos) [cite: 3] | Numérico |
| ELAPSED_TIME | Tempo total real de voo (em minutos) [cite: 3] | Numérico |
| AIR_TIME | Tempo no ar (em minutos) [cite: 3] | Numérico |
| DISTANCE | Distância entre origem e destino (em milhas) [cite: 3] | Numérico |
| WHEELS_ON | Horário em que as rodas tocaram o solo (HHMM) [cite: 3] | Inteiro |
| TAXI_IN | Tempo taxiando até o portão de desembarque (em minutos) [cite: 3] | Numérico |
| SCHEDULED_ARRIVAL | Horário de chegada programado (HHMM) [cite: 3] | Inteiro |
| ARRIVAL_TIME | Horário de chegada real (HHMM) [cite: 3] | Inteiro |
| ARRIVAL_DELAY | Atraso na chegada (em minutos) [cite: 3] | Numérico |
| DIVERTED | Indica se o voo foi desviado (1 = sim, 0 = não) [cite: 3] | Binária |
| CANCELLED | Indica se o voo foi cancelado (1 = sim, 0 = não) [cite: 3] | Binária |
| CANCELLATION_REASON | Motivo do cancelamento (A=Airline, B=Weather, C=NAS, D=Security) [cite: 3] | Categórica |
| AIR_SYSTEM_DELAY | Atraso causado por controle de tráfego aéreo [cite: 3] | Numérico |
| SECURITY_DELAY | Atraso causado por problemas de segurança [cite: 3] | Numérico |
| AIRLINE_DELAY | Atraso causado pela companhia aérea [cite: 3] | Numérico |
| LATE_AIRCRAFT_DELAY | Atraso causado por chegada tardia da aeronave [cite: 3] | Numérico |
| WEATHER_DELAY | Atraso causado por condições meteorológicas [cite: 3] | Numérico |
