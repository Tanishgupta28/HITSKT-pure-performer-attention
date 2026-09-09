# HiTSKT fixed action-window audit

Status: consequential representation decision required before tensor/window
generation.

The legacy HiTSKT loader reserves one position for EOS and retains only the
most recent `action_size - 1` interactions in each session. The published
commands use `action_size=64` for ASSIST2017 and `action_size=32` for Junyi and
EdNet, giving capacities of 63 and 31 real interactions respectively.

Applying that behavior to the validated full event stores would discard:

| Dataset | Sessions | Capacity | Sessions over capacity | Share over capacity | Interactions discarded | Share discarded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ASSIST2017 | 12,402 | 63 | 5,329 | 42.968876% | 303,971 | 34.334009% |
| Junyi | 600,154 | 31 | 146,357 | 24.386574% | 4,310,593 | 29.403337% |
| EdNet-KT1 | 2,577,988 | 31 | 666,692 | 25.860943% | 35,198,097 | 42.955485% |

Maximum session lengths are 938 (ASSIST2017), 3,924 (Junyi), and 13,080
(EdNet-KT1). Median lengths are 55, 16, and 18 respectively.

This is not a minor padding detail: legacy truncation would remove 29–43% of
the retained supervised interactions and make target coverage differ materially
across models. Alternatives such as treating fixed-size chunks as sessions
would change the meaning of the approved 10-hour session hierarchy. A decision
is therefore required before the shared rolling-history representation is
generated.
