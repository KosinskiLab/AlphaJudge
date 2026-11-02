# Third-party licensing

## Scope of the MIT license

The top-level `LICENSE` applies to original AlphaJudge code and contributions
for which the project can grant permission. Third-party code and data retain
their own terms. In particular, MIT does not replace the rights of the authors
identified in the source files below.

## CCP4-derived code and tables: permission review required

The following provenance statements are present in the source as of the
licensing review on 2026-11-02. They require checking the exact upstream source
versions and applicable permissions before licensing the complete distribution.

| File in `src/alphajudge/biophysics/` | Recorded provenance |
| --- | --- |
| `connolly.py` | Described as a Python translation of the `sc.f` MDS subroutine. The header credits copyright 1986 Michael L. Connolly and copyright 1999-2004 Michael Lawrence. The radii table is described as taken from `sc_radii.lib`. |
| `sc.py` | Described as ported from SCASA (Lawrence & Colman, 1993; Connolly, 1983). |
| `prosurf.py` | `calcSphericalCode` is described as a port from CCP4 PISA. |
| `pisa_molref.py` | Values are described as extracted from CCP4 PISA's MolRef table. |
| `srs_chemistry.py` | Tables are described as mirroring the compiled CCP4SRS database shipped with CCP4 7.1. |
| `bonds.py` | Implements CCP4SRS/PISA-like bond scoring and uses constants attributed to CCP4SRS; review alongside the chemistry tables. |

The [CCP4 SC documentation](https://www.ccp4.ac.uk/html/sc.html) states that SC
includes a modified Connolly MDS routine supplied with the author's consent.
That statement does not establish permission to relicense a Python translation
under MIT. CCP4 distributes components under different terms; its
[GNU software listing](https://www.ccp4.ac.uk/licensing/GPL_and_LGPL_software.html)
does not establish a blanket license for these particular routines or tables.

These findings identify material needing review, not a legal determination that
every listed algorithm, numerical value or implementation is copyright-restricted.
Existing source attributions are preserved. No separate licensing permissions
for this material were available when this update was prepared.

## AlphaFold fixtures

The following directories carry their own `TERMS_OF_USE.md`:

- `test/fixtures/af3x_4g3y_bc_dsso/`
- `test_data/af3/neg_dimers/Q14974+Q13033/`
- `test_data/af3/pos_dimers/Q9BUL8+Q13033/`

The three terms files are byte-identical at the reviewed revision. A verbatim
[copy](licenses/AlphaFold-fixtures-TERMS_OF_USE.md) is included in package license
files so those terms are not mistaken for MIT. This copy does not establish that
the fixtures were obtained or redistributed in compliance with the terms;
confirm their provenance before including them in any release archive.

Third-party dependencies and tools, including AlphaFold, retain their own
licenses. AlphaJudge's license does not alter restrictions on its input data
or model outputs.

## Distribution metadata and outstanding work

The package deliberately omits a distribution-wide SPDX `license` expression
and the old MIT classifier while the CCP4 provenance review is outstanding.
`license-files` includes the project license, this inventory and the fixture
terms in both wheels and source distributions. The README identifies original
AlphaJudge code as MIT without describing the entire archive as MIT-only.

Before publishing a release with the revised licensing:

1. Identify the upstream versions and licenses for the routines and tables above.
2. Obtain any needed permissions, comply with the applicable upstream terms, or
   replace affected material with an independently developed implementation.
3. Preserve the required copyright notices and license texts, and confirm fixture
   redistribution terms for any data included in the release.
4. Set the package SPDX expression to the licenses actually applicable to the
   resolved archive and verify the built wheel and source distribution.

This inventory does not grant missing third-party permissions or represent the
complete distribution as cleared for unrestricted redistribution.
