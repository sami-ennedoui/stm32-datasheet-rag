#!/usr/bin/env bash
# Download the STM32H7 reference manual RM0433 (real public PDF).
# RM0433 covers STM32H742, STM32H743/753 and STM32H750. The N7 Racing Team
# uses STM32H7 boards, so this is the right document.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${HERE}/data/rm0433.pdf"
mkdir -p "${HERE}/data"

# Primary source: STMicroelectronics.
ST_URL="https://www.st.com/resource/en/reference_manual/rm0433-stm32h742-stm32h743753-and-stm32h750-value-line-advanced-armbased-32bit-mcus-stmicroelectronics.pdf"
# Public university mirror (used if st.com blocks the request).
MIRROR_URL="https://ucilnica.fri.uni-lj.si/pluginfile.php/216967/mod_folder/content/0/rm0433-stm32h742-stm32h743753-and-stm32h750-value-line-advanced-armbased-32bit-mcus-stmicroelectronics.pdf?forcedownload=1"

UA="Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"

echo "Downloading RM0433 ..."
if curl -fSL --http1.1 -A "${UA}" --max-time 240 -o "${OUT}" "${ST_URL}"; then
  echo "Got it from st.com."
else
  echo "st.com failed, trying public mirror ..."
  curl -fSL --http1.1 -A "${UA}" --max-time 240 -o "${OUT}" "${MIRROR_URL}"
fi

echo "Saved to ${OUT}"
ls -lh "${OUT}"
