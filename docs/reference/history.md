# 🧭 Background & context

## 🧪 Early models

The basic recombination–diffusion–permeation picture originates from early work on
atomic hydrogen interaction with metals, in particular:

- Ali-Khan, I.; Dietz, K. J.; Weldebrock, F. G.; Wienhold, P.,  
  *The rate of hydrogen release out of clean metallic surfaces*.  
  [DOI: 10.1016/0022-3115(78)90167-8](https://doi.org/10.1016/0022-3115(78)90167-8)

- Pick, M. A.; Sonnenberg, K.,  
  *A model for atomic hydrogen–metal interactions—application to recycling, recombination and permeation*.  
  [DOI: 10.1016/0022-3115(85)90459-3](https://doi.org/10.1016/0022-3115(85)90459-3)

These works established the surface–bulk coupling via recombination that is still
commonly used today.  
This package uses only the **simplest form** of these ideas.

---

## 📚 Formal theory

A. A. Pisarev and his student E. D. Marenkov developed this framework much further,
publishing multiple papers and textbooks that treat hydrogen transport in a
systematic, physically detailed way.

Their work is **not followed directly** here.  
It is referenced mainly for **context and future reading**, to show how far the topic
extends beyond the simplified model implemented in this code.

---

## 🔍 Modern checks

More recent work revisits recombination coefficients and their consistent use in
transport calculations, for example:

- Schmid, K.; Zibrov, M.,  
  *On the use of recombination rate coefficients in hydrogen transport calculations*.  
  [DOI: 10.1088/1741-4326/ac07b2](https://doi.org/10.1088/1741-4326/ac07b2)

Such papers are useful for understanding limitations and ambiguities of simplified
boundary conditions, rather than as direct implementation guides.

---

## 🧩 This code

This is a **tiny, intentionally simple** codebase.

That is a feature, not a limitation:
- easy to understand,
- easy to modify,
- suitable for teaching, testing ideas, or quick estimates.

For **production-level simulations**, established tools are usually a better choice:

- **FESTIM** — https://festim.readthedocs.io/en/latest/  
- **TMAP8** — https://mooseframework.inl.gov/TMAP8/

With these tools now open source, developing a custom Monte Carlo model is also much
more approachable than it used to be.

This package is best viewed as an **entry point**, not an endpoint 🚪
