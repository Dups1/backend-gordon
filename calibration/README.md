# Protocolo de calibración Gordon

La versión `provisional-en-US-v1` no debe promoverse a evaluación automática
estable hasta completar este protocolo.

## Analizador offline

El repositorio incluye una utilidad sin servicios externos ni dependencias
adicionales. Cada línea del JSONL debe ser una evaluación con su etiqueta humana
final (consenso o adjudicación):

```json
{"assessmentId":"assessment-1","speakerId":"speaker-1","labelSource":"human","humanRatingStatus":"consensus","raterCount":2,"split":"train","mode":"reading","cefr":"B1","subgroups":{"l1":"es","device":"mobile","condition":"quiet"},"dimensions":{"communication":{"automaticScore":72.5,"humanBand":3,"raterBands":[3,3]},"pronunciation":{"automaticScore":68,"humanBand":3,"raterBands":[3,3]},"grammar":{"automaticScore":75,"humanBand":3,"raterBands":[3,3]},"vocabulary":{"automaticScore":70,"humanBand":3,"raterBands":[3,3]},"fluency":{"automaticScore":64,"humanBand":2,"raterBands":[2,3]}}}
```

- `split`: `train`, `validation` o `test`.
- `automaticScore`: puntuación Gordon entre 0 y 100.
- `humanBand`: banda humana entera entre 0 y 4.
- `labelSource`: debe ser `human`; el analizador rechaza GPT-OSS, Qwen u otra
  fuente automática como verdad de calibración.
- `humanRatingStatus`: `consensus` o `adjudicated`, con `raterCount >= 2`.
- `raterBands`: bandas individuales de los docentes; permiten calcular kappa
  humano antes de usar `humanBand` como etiqueta final.
- `subgroups`: valores escalares pseudonimizados; `mode` y `cefr` se incluyen
  automáticamente en el resumen.
- Una dimensión ausente no se imputa: reduce la cobertura de esa dimensión.
- Un mismo `speakerId` puede tener varias evaluaciones dentro de un split, pero
  si aparece en más de un split el análisis falla para impedir fuga de datos.

Ejecuta:

```bash
npm run calibration:analyze -- ./ruta/dataset.jsonl > reporte.json
```

El reporte contiene MAE y Spearman en escala 0–100, kappa cuadrático
modelo-humano y kappa humano entre los dos primeros evaluadores. La banda
automática es el `automaticScore` redondeado a la banda de 25 puntos más
cercana. También incluye cobertura y métricas por dimensión, split y cada
valor de subgrupo. Spearman o kappa se devuelven como `null` cuando la muestra
o su variación no permiten calcularlos.

## Piloto

- 600 sesiones balanceadas, 100 por CEFR A1–C2.
- Cada sesión aporta lectura guiada y respuesta espontánea: 1,200 audios.
- Adultos/universitarios, diversidad de L1/acento, dispositivo, ruido y
  condiciones de grabación.
- Dos docentes puntúan independientemente las cinco dimensiones en bandas 0–4.
- Diferencias mayores a una banda pasan a adjudicación.
- Split por hablante: 60 % train, 20 % validation y 20 % test bloqueado.
- Ningún hablante puede aparecer en más de un split.

## Gates

- Kappa humano ponderado >= 0.70 antes de declarar etiquetas gold.
- MAE automático <= 8/100.
- Spearman >= 0.75.
- Kappa modelo-humano >= 0.65.
- Evidencia/citas inválidas < 1 %.
- Brecha de MAE entre subgrupos <= 5 puntos.

Se medirán además cobertura, tasa de abstención, ancho de intervalos,
repetibilidad por proveedor y correcciones docentes.

## Qwen

Qwen3.6-27B se entrenará solamente después de reunir 5,000 respuestas distintas
con etiquetas humanas. GPT-OSS es baseline, no ground truth. Qwen se ejecutará
en sombra y solo se promoverá si reduce MAE al menos 1.5 puntos con bootstrap
pareado, sin regresión mayor a 2 puntos por subgrupo ni aumento de evidencia
inventada. Los scores acústicos de Azure nunca serán modificados por el LLM.
