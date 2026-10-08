# Lumen design and motion

The refreshed interface uses larger, locally hosted typography, higher-contrast
supporting text, larger Lucide icons, generous controls, green and lilac surfaces,
and a dimensional sprout identity. It applies to account entry, all workspace
tabs, lessons, assessments and completion feedback.

## Typography and interaction

- DM Sans variable font for body text; Plus Jakarta Sans variable font for
  headings. The fonts are bundled through `next/font/local`; no browser request
  to a Google Fonts CDN is required.
- Main copy and inputs are16px, desktop navigation16px, navigation icons23px,
  primary controls at least52px high. Smaller metadata is visually subordinate.
- Desktop headings scale between32–50px. Mobile pages retain readable text and
  controls, with a navigation drawer below1000px rather than a compressed sidebar.
- Cards, navigation, selected answers, progress bars, feedback, lessons and tab
  entry use short transitions. Hover and focus styles indicate actionable items.

`frontend/src/app/globals.css` contains the base layouts and shared type scale;
`frontend/src/app/redesign.css` contains the refreshed visual system.

## Logo and motion

`BrandMark` is an SVG sprout on a layered CSS tile with perspective, highlights,
shading and extrusion. The large hero artwork gently floats. It does not require
a WebGL canvas or a3D engine. The favicon follows the same dimensional identity.

`frontend/src/assets/lumen-orbit.json` is original Lumen animation data, authored
for this project. Six particles follow simple orbits around the sculpture.
`GrowthScene` lazy-loads pinned `lottie-web@5.13.0` after hydration, renders SVG,
destroys the player on unmount, and pauses it outside the viewport or in a hidden
tab. The static sculpture remains available if the animation cannot load.

The global **Motion on/off** control persists a preference in browser local
storage. It stops both CSS animations/transitions and Lottie playback. System
`prefers-reduced-motion` takes precedence, renders a still frame and removes the
motion toggle. Entry animations use opacity and transforms, keeping layout stable.
All decorative artwork is hidden from assistive technology.

## Asset provenance

Fonts are distributed under the SIL Open Font License1.1. Exact license files
are bundled next to the assets in `frontend/src/app/fonts/`.

| Asset | Official source | SHA256 |
| --- | --- | --- |
| DM Sans variable TTF | [Google Fonts](https://github.com/google/fonts/tree/main/ofl/dmsans) | `8cd08d97e89c24d0aa92edd2f0f4c8ee6195eee9b7c9f154865a58b02f0c1c0d` |
| Plus Jakarta Sans variable TTF | [Google Fonts](https://github.com/google/fonts/tree/main/ofl/plusjakartasans) | `89b3fb38aa0d275d7a731d0d817a4f1622b316b4d7fbdedcf02ee9099ff68bc8` |
| Original orbital animation | `frontend/src/assets/lumen-orbit.json` | Project-authored; tracked in Git |

Player reference: [official Lottie load options](https://github.com/airbnb/lottie-web/wiki/loadAnimation-options).

## Validation

`frontend/tests/design.spec.ts` checks loaded local fonts and actual type/icon
sizes, rendered Lottie artwork, persistent pause controls, reduced motion,
responsive widths from360px to1440px and absence of external font requests.
The existing complete learning-cycle browser test also covers profile editing,
diagnostic/resume, progress, lessons and mobile practice. Final milestone evidence
is recorded in `reports/platform/design_validation.json` and indexed in
`project-memory/PLATFORM_STATE.md`.
