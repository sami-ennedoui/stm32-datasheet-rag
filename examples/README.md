# Example run, real output

This folder holds real output captured from a working run on Python 3.12, with
the real RM0433 reference manual ingested (3353 pages, 6608 chunks) and the
Hugging Face inference API answering.

Files:

- `health.json`: the `/health` response.
- `ask_usart_baudrate.json`: the real `/ask` response for the question below.
- `draft_code_usart.json`: the real `/draft-code` (bonus) response.
- `ingest.log`: the ingest run log.

## Question

> How is the USART baud rate configured on STM32H7?

## Answer returned by POST /ask

The USART baud rate on STM32H7 is configured using the USART_BRR register
[page 2085]. This register can only be written when the USART is disabled
(UE = 0) [page 2085]. The baud rate is defined by the bits BRR[15:0] in the
USART_BRR register [page 2085]. The register may also be automatically updated
by hardware during auto baud rate detection [page 2085].

## Citations returned

| page | chunk_id | score |
|------|----------|-------|
| 2077 | p2077-2  | 0.7392 |
| 2040 | p2040-0  | 0.6938 |
| 2039 | p2039-0  | 0.6914 |
| 2085 | p2085-0  | 0.6825 |
| 2031 | p2031-0  | 0.6684 |

Page 2085 is section 48.8.5, the USART baud rate register (USART_BRR), in the
real RM0433. The cited pages are all from the USART/UART chapter, which confirms
retrieval landed on the right part of the manual.

## Bonus, POST /draft-code

The same question family fed to `/draft-code` produced a small C snippet that
disables USART1, writes USART1->BRR, and re-enables it, with page citations.
See `draft_code_usart.json`.

To reproduce, follow the steps in the top level README, then run:

```bash
scripts/ask.sh "How is the USART baud rate configured on STM32H7?"
```
