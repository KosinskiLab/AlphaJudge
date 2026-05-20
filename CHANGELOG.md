# Changelog

## 1.0.2 - 2026-05-20

- Added official DeepMind AF3 layout and Boltz-2 parser support; custom parser subclasses should implement `BaseParser.detect` as `detect(d: Path) -> bool` because it is now a static method.
