"""Prompt builder for the My3DFigure direct GPT Image 2 flow."""

import random


WARDROBE_COMPLETIONS = {
    "shorts": "separate garments with tailored shorts",
    "long_trousers": "separate garments with full-length straight-leg trousers",
    "wide_leg_trousers": "separate garments with fluid wide-leg trousers",
    "jeans": "separate garments with straight-leg or relaxed jeans",
    "cropped_trousers": "separate garments with cropped ankle-length trousers",
    "skirt": "separate garments with a knee-length skirt",
    "short_skirt": "separate garments with an above-knee A-line or pleated skirt",
    "long_skirt": "separate garments with a flowing ankle-length skirt",
    "dress": "a coordinated knee-length dress",
    "short_dress": "a coordinated above-knee dress",
    "long_dress": "a coordinated flowing ankle-length dress",
    "jumpsuit": "a one-piece full-length jumpsuit",
    "romper": "a one-piece short-leg romper",
    "bodysuit": "a coordinated bodysuit with a skirt or tailored shorts",
}

MY3D_PET_PROMPT_VERSION = 'my3d-pet-v6-shared-policies'

WARDROBE_PALETTES = {
    "pastel": "soft pastel color-blocking with two or three harmonious colors",
    "bright": "a lively but balanced palette with a clear accent color",
    "earthy": "warm earthy tones with textured natural fabrics",
    "denim": "denim blue paired with a contrasting warm or light accent",
    "nautical": "navy, cream, and one small red or yellow accent",
    "floral": "a subtle floral or botanical accent balanced with solid colors",
    "retro": "a tasteful retro color combination with one small pattern detail",
    "minimal": "a refined palette with varied texture and one accent color",
    "jewel": "balanced jewel tones with a restrained contrasting accent",
    "ocean": "teal and blue balanced with a warm light accent",
    "sunset": "warm coral and saffron balanced with cream",
    "berry": "berry and dusty rose balanced with a cool dark accent",
}

UPPER_GARMENT_COMPLETIONS = {
    "tee": "a textured or color-block T-shirt",
    "blouse": "a softly draped blouse",
    "shirt": "a tailored casual shirt",
    "knit": "a textured knit top",
    "cardigan": "a cardigan with a coordinated inner top",
    "sweatshirt": "a casual sweatshirt",
    "hoodie": "a casual hoodie",
    "vest": "a knit vest over a coordinated inner top",
    "jacket": "a light casual jacket with an inner top",
    "sleeveless": "a coordinated sleeveless top",
}

FOOTWEAR_STYLES = {
    "sneakers": "color-accent athletic or retro sneakers",
    "canvas": "two-tone canvas shoes or canvas slip-ons",
    "sandals": "coordinated strappy or sporty sandals",
    "loafers": "textured or two-tone loafers",
    "mary_janes": "color-accent Mary Jane shoes",
    "ballet_flats": "coordinated ballet flats",
    "ankle_boots": "colored leather or suede ankle boots",
    "tall_boots": "coordinated knee-high boots",
    "espadrilles": "textured espadrilles with a colored upper",
    "dress_shoes": "coordinated dress shoes with contrasting material details",
}

FOOTWEAR_COMPLETIONS = {
    "shorts": "coordinated canvas sneakers, sporty sandals, or colorful slip-on shoes",
    "skirt": "coordinated Mary Jane shoes, ankle boots, ballet flats, or refined sneakers",
    "dress": "coordinated ballet flats, ankle-strap shoes, Mary Jane shoes, or elegant ankle boots",
    "cropped_trousers": "coordinated loafers, retro trainers, ankle boots, or textured slip-on shoes",
    "jumpsuit": "two-tone loafers, retro trainers, ankle boots, or canvas sneakers",
    "romper": "color-accent sandals, Mary Jane shoes, espadrilles, or slip-on shoes",
    "bodysuit": "ballet flats, ankle-strap shoes, colorful sneakers, or Mary Jane shoes",
}

# These are concrete colors for unknown areas, never replacements for visible
# source colors. Keep the shoe accent independent of a dark source neckline.
UNSEEN_COLOR_DIRECTIONS = {
    "pastel": "dusty lilac or sage for unseen clothing; blush or cream shoes with a colored accent",
    "bright": "cobalt blue or coral for unseen clothing; cream shoes with a contrasting colored accent",
    "earthy": "terracotta or olive for unseen clothing; tan/cognac shoes with cream details",
    "denim": "medium-blue denim for unseen clothing; burgundy or mustard shoe accents",
    "nautical": "cream or navy for unseen clothing; red or yellow accents on the shoes",
    "floral": "a small botanical pattern on muted green or rose unseen clothing; solid sage or rose shoes",
    "retro": "rust or teal for unseen clothing; two-tone cream and brown shoes",
    "minimal": "stone, taupe or muted blue for unseen clothing; off-white shoes with a blue accent",
    "jewel": "emerald or plum for unseen clothing; cream shoes with a plum or gold-tone accent",
    "ocean": "teal or cobalt for unseen clothing; cream or caramel shoes with a teal accent",
    "sunset": "coral or saffron for unseen clothing; cream shoes with a warm colored accent",
    "berry": "wine or dusty rose for unseen clothing; navy or mauve shoes with a light accent",
}


def _build_my3d_direct_prompt_legacy(
    wardrobe_profile: str | None = None,
    wardrobe_palette: str | None = None,
) -> str:
    """Return the fixed output requirements without any external image analysis."""
    wardrobe_completion = WARDROBE_COMPLETIONS.get(wardrobe_profile or "", WARDROBE_COMPLETIONS["shorts"])
    palette_direction = WARDROBE_PALETTES.get(wardrobe_palette or "", WARDROBE_PALETTES["pastel"])
    footwear_completion = FOOTWEAR_COMPLETIONS.get(
        wardrobe_profile or "", FOOTWEAR_COMPLETIONS["shorts"]
    )
    return "\n".join(
        [
            "OUTPUT REQUIREMENTS (highest priority):",
            "Create exactly one polished cute 3D chibi figurine of the single primary person in the uploaded photo.",
            "ABSOLUTE QUALITY PRIORITY: produce correct anatomy without deleting clear source evidence. Clearly visible hand poses, phones, cigarettes, carried items, accessories, and readable tattoos are required source features, not optional complexity. Prevent malformed anatomy by using natural finger occlusion and clean object separation while preserving the source action; never solve an anatomy problem by changing a clearly visible holding hand into an empty, lowered, or unrelated pose. Only genuinely unreadable tiny details may be omitted.",
            "Portrait 2:3 canvas, pure white seamless studio background, soft floor shadow only.",
            "NON-NEGOTIABLE PERSON IDENTITY RULE: preserve the primary person's apparent gender presentation and age presentation exactly. Never turn a woman or girl into a man or boy, and never turn a man or boy into a woman or girl. Preserve the visible face identity, face shape, hair length, hair color, hairline, and recognizable hairstyle; stylize them into 3D chibi form without replacing them.",
            "NON-NEGOTIABLE HEAD ORIENTATION RULE: never introduce an anatomical head tilt that is absent from the source. Judge head tilt only RELATIVE TO THE PERSON'S OWN NECK, SHOULDER LINE, CHEST CENTERLINE, AND TORSO, never relative to the rectangular image border, screen horizontal, scenery, furniture, or camera horizon. First distinguish global photo/camera roll from a real head-to-body tilt. If the face, neck, shoulders, and torso share approximately the same screen-space rotation, the whole photo is merely rotated: mentally rotate the complete person upright before interpreting the pose, render the body upright on the output canvas, and keep the head aligned naturally above the neck with no left or right lean. Do not transfer global camera roll into head roll after straightening the body. Only preserve a head tilt when the head axis visibly deviates from the neck and torso in the source itself. If this distinction is uncertain, default to an upright head.",
            "For a headshot showing mainly the head and possibly the neck, shoulders, or a small part of the upper chest, always set head roll to exactly 0 degrees: the line through both pupils is horizontal, the nose bridge and neck centerline are vertical, the chin is centered above the sternum, and the top-of-head midpoint is directly above the neck. The head must not lean even slightly toward the viewer's left or right. Hair asymmetry, facial asymmetry, body stance, clothing, cuteness, fashion styling, black bars, screenshots, and a rotated camera must never rotate or slant the head. For a genuine half-body or full-body source, preserve the source's HEAD-TO-BODY RELATIONSHIP rather than its raw angle inside the image: a head aligned with the source torso remains aligned in the result; only a head visibly tilted relative to that torso may remain tilted. This rule overrides all creative pose choices.",
            "GENERAL VISIBLE-DETAIL PRESERVATION: apply this rule to every uploaded photo, not to any one example. Before rendering, inventory all clearly identifiable person-owned details visible on the primary person: headwear and hair ornaments; eyewear and jewelry; tattoos and body markings; garment colors, prints, fasteners, and patches; watches and wristwear; bags, straps, charms, and keychains; and anything clearly held, hugged, worn, clipped, attached, carried, or supported. Preserve each detail's category, count, dominant color, approximate scale, anatomical side, nearby body or garment landmark, and interaction with the correct hand, limb, clothing, or strap. A detail remains required wherever it appears on the body or outfit; the examples in these instructions are illustrative, never an exhaustive list. Do not remove a clear detail to simplify the design, change the pose, or avoid a difficult grip. Only a genuinely unreadable, ambiguous, or unowned tiny feature may be omitted.",
            "NON-NEGOTIABLE HEADWEAR RULE: if the primary person is clearly wearing anything on the head in the source, preserve it as part of the character. This includes a party crown, birthday hat, cap, beanie, headband, hair clip, bow, or other visible head ornament. Match its visible shape, material, dominant colors, patterns, placement, tilt, and attachment to the head; do not remove, replace, or reinterpret it as hair. If an adult's hand is touching or adjusting the headwear, remove only that external hand while keeping the headwear intact. If no headwear is clearly present in the source, do not add any.",
            "NON-NEGOTIABLE ACCESSORY GATE: inspect the source before rendering any accessory. An accessory may appear only when it is clearly and unambiguously visible as worn by or physically attached to the primary person, their clothing, or their carried item in the source. Preserve clearly visible eyewear, earrings, necklaces and pendants, bracelets, watches, rings, hair ornaments, badges, brooches, lanyards, belt details, keychains, bag charms, and similar small attached details. If the source face has no eyewear, render a completely unobstructed face with no glasses or sunglasses. If no wristwear is clearly visible, render bare wrists with no watch or bracelet. Never add accessories as a fashion choice, and never interpret reflections, shadows, hair, scenery, or cropped/hidden areas as accessories. Clothing design freedom applies only to garments and footwear, never to accessories.",
            "NON-NEGOTIABLE CARRIED-ITEM GATE: preserve every clearly visible item that the primary person is unmistakably holding, hugging, carrying, wearing, or supporting, such as a water bottle, cup, phone, camera, keys, card, pen, lighter, snack, flower, doll, plush toy, handbag, shoulder bag, backpack, or a cigarette visibly held between the fingers. A bottle visibly resting against the person's shoulder or back, a bag connected by a visible strap, or a small object clearly pinched by the primary person's fingers counts as carried even when part of the hand is occluded. Reproduce its item type, count, dominant color, approximate size, screen-left or screen-right position, carrying method, strap placement, and contact with the correct hand, arm, shoulder, torso, or fingers. A carried item is part of the source pose and must not be deleted merely because it is a prop or is visually small. Never add an item that is absent, belongs to another person, merely appears in the background, or has uncertain ownership.",
            "SMALL-ITEM FIDELITY: inspect both hands, finger gaps, wrists, clothing, pockets, straps, and bags. Preserve every clearly identifiable person-owned item, including a phone and cigarette when both are visible, with the correct count, hand, position, and attachment. Keep objects separate from skin. Omit only an item whose identity or ownership is genuinely unreadable; never omit a clear item merely because the grip is complex, and never convert an item into a finger or body part.",
            "CIGARETTE FIDELITY RULE: when a cigarette is clearly visible, preserve exactly one in the same hand and approximately the same direction, between the intended fingers. Build a valid hand around it using natural occlusion: only the visible gripping fingers need to be exposed, while the other anatomical fingers may remain naturally curled or hidden. The cigarette must remain separate from skin and must not become, split into, or create a finger. Show smoke, ash, or a glowing tip only when clearly visible. If no cigarette is visible, never add one.",
            "TATTOO AND BODY-MARKING FIDELITY: preserve every clearly readable tattoo or distinctive body marking anywhere on the primary person. Treat each readable marking as required source evidence even when it occupies a small area. Anchor it to the same anatomical side and the same nearby body, joint, garment-edge, or neckline landmark. Match its approximate size, orientation, colors, and motif. Do not cover a required marking by redesigning clothing, moving hair, changing a strap, adding an accessory, or changing the pose; do not relocate, enlarge, or invent it. Only a marking genuinely too blurred, obscured, cropped, or tiny to identify at all may be omitted.",
            "Use neutral white studio color balance. Preserve the primary person's visible skin-tone depth and undertone, but neutralize color cast from outdoor sunlight, scenery, clothing reflection, or camera white balance. Do not make skin yellow, orange, sallow, or over-saturated.",
            "Preserve the primary person's visible hairstyle, facial direction, and only those worn accessories that pass the strict accessory gate above.",
            FACE_DIRECTION_POLICY,
            "CLOTHING EVIDENCE ORDER: inspect the source photo and apply only the first matching rule below.",
            "1. If the full outfit is visible, reproduce every visible garment, color, silhouette, and footwear faithfully; do not redesign the clothing.",
            "2. If a dress is clearly visible, preserve the dress's visible color and silhouette and extend it naturally below the crop; do not replace it with separate garments.",
            "3. If any part of a lower garment is visible, it is source evidence and must be preserved and extended naturally. First locate the upper garment's lower hem and distinguish the trousers, shorts, skirt, or dress fabric visible below it. Preserve the visible lower-garment category, dominant color, pattern, material, fit, waistband or belt, pocket and seam details, and apparent length direction. Do not redesign it merely because only a small section is visible.",
            "PARTIAL LOWER-GARMENT VISIBILITY RULE: dim lighting, yellow or colored illumination, haze, shadow, low contrast, bottom-edge fading, obstruction, and partial cropping do not make visible trousers or other lower garments 'missing'. Correct the lighting cast while retaining the underlying garment color and construction. Treat the lower garment as freely designable only when no usable part of it is visible anywhere in the source.",
            "SHORT-BOTTOM DISAMBIGUATION: when the source shows only the waistband and upper thigh area of a lower garment, with its hem or full legs cropped out, do not automatically extend it into full-length trousers. Use visible pocket placement, rivets, fasteners, waistband height, side seams, and silhouette to distinguish shorts, a skirt, skort, or trousers. If the evidence supports a short bottom or remains genuinely ambiguous, preserve the visible short-bottom construction and complete it as an age-appropriate shorts, skirt, or skort; create long trousers only when clear long-leg lines or a visible trouser hem support them.",
            "4. If the upper garment's style is clearly visible and no usable part of the lower garment is visible, faithfully preserve the visible upper garment's style and color. Design only the genuinely missing lower garment and shoes as a coordinated match.",
            "VISIBLE UPPER-CLOTHING FIDELITY: when any upper clothing is visible, preserve its actual garment category and construction, including outerwear versus shirt, open versus closed front, layered inner top, neckline, collar, sleeve length, fasteners, dominant colors, and overall silhouette. Match every clearly visible upper-garment color; for example, never change a visible beige jacket into a blue shirt. Do not replace a visible jacket or layered outfit with a different blouse, sweater, or shirt. Creative clothing variation is allowed only in genuinely unseen or indeterminate areas.",
            "5. If only an upper-garment color is visible but its style is unclear, preserve that visible color while freely designing a richer, coordinated upper-garment style. Design the missing lower garment and shoes to match.",
            f"6. If the source is only a headshot with no usable clothing information, freely create one polished, coordinated full outfit: {wardrobe_completion}.",
            f"For clothing areas that are free to design, use this rotating style direction: {palette_direction}.",
            "For every missing clothing area, use a tasteful varied color palette, harmonious fabrics, layered or patterned details when appropriate, and a complete coherent silhouette. Do not default to plain long trousers, an all-black bodysuit, or an all-white blouse-and-trousers outfit unless the source photo visibly requires it.",
            "FOOTWEAR EVIDENCE RULE: if the source clearly shows the shoes, reproduce their visible type, construction, colors, patterns, and fastening faithfully. If the shoes are cropped out, hidden, or too unclear to identify, design them freely instead of guessing that the source shoes were plain white sneakers.",
            f"For freely designed footwear, vary the result among these age-appropriate directions: {footwear_completion}. Coordinate the shoe colors and materials with the completed outfit and rotating palette, while giving the shoes a visible complementary or accent color. Do not repeatedly default to the same white low-top sneaker design.",
            "Show the complete figure from top of hair through both shoes; nothing may be cropped.",
            "Place the figure at the exact horizontal center and visual vertical center.",
            "The complete figure must occupy about 78-82% of the canvas height, leaving even clean margins.",
            "Use a refined collectible chibi proportion, never a bobblehead proportion: the complete head including hair should occupy about 36-40% of the figure height.",
            "Balance the silhouette with adequately broad shoulders, a substantial torso, natural arm length, and proportionate legs and shoes. Do not make the head visually wider or heavier than the whole body beneath it.",
            "Exactly two arms, two hands, two legs, two feet, and two shoes; anatomically coherent joints.",
            "NON-NEGOTIABLE HAND ANATOMY: each hand has one palm and exactly five anatomical digits: one thumb plus four fingers. Preserve the source hand's position, wrist angle, gesture, and holding action. For a complex grip, show only the fingers that would naturally be visible and let the remaining fingers stay correctly curled or occluded; do not fan out extra fingertips. No sixth, duplicated, detached, floating, fused, melted, or tangled digit, no second palm, and no fragment of another person's hand. Build the valid hand first and place each required held item as separate non-skin geometry against only the intended fingers. Do not delete a clearly visible phone, cigarette, or other held item and do not lower or replace the hand merely to simplify anatomy.",
            "Remove all source scenery, other people and their body parts, background objects, nearby loose objects, and items with uncertain ownership. Do not depict, name, or reinterpret removed items.",
            "Do not invent props, bottles, bags, phones, toys, straps, containers, other people, or external hands. Clearly carried items that pass the carried-item gate above are required exceptions and must be preserved.",
            "Do not invent any accessory that is not visibly worn by the primary person.",
            "Preserve the reference person's head direction, torso rotation, gestures, wrist rotation, hand-facing direction, and each hand's height relative to the face and torso. Do not mirror or swap left and right.",
            "If the source crop does not clearly show an arm or hand pose, complete both arms in a simple neutral pose hanging naturally at the sides with empty relaxed hands. When the source clearly shows a hand or arm holding, hugging, carrying, or supporting an item, preserve that action and the item instead. Never invent a hand touching the face, hair, head, an accessory, or an item when that contact is not clearly visible in the source.",
            "Only if a watch is clearly visible on the source wrist, use its watch face to identify the back-of-wrist side and reconstruct the hand accordingly; otherwise keep the wrist bare.",
            "FINAL MANDATORY QUALITY CHECK BEFORE RENDERING: (1) exactly one person, two arms, two hands, two legs, two feet, and no stray fragments; (2) each hand has one palm and five anatomical digits total, with no extra, duplicated, detached, fused, melted, or tangled fingers; (3) compare the result against the complete visible-detail inventory and confirm every clear worn, held, attached, carried, clothing, accessory, marking, and pose detail remains at the correct side, location, count, and contact point; (4) the head remains aligned with the source torso after global photo rotation is removed. Correct anatomy without deleting clear source evidence. Omission is acceptable only for a genuinely unreadable tiny marking or ambiguous object.",
            "When an arm originally exits the crop into another person's hand, completely erase the other person's entire hand and every one of its fingers, then reconstruct only the primary person's single continuous hand with one palm, one thumb, and four fingers following the original reach. Do not blend the two hands, leave a disconnected digit, make a fist, or add a held object.",
            "Only if the source clearly shows both eyewear and a hand touching it, preserve gentle fingertip contact at the outer frame or temple and keep all shapes separate. If either the eyewear or that contact is absent, do not create eyewear or an eyewear-touching gesture.",
            "Render the person as a high-quality collectible 3D chibi character while preserving these requirements.",
        ]
    )


MY3D_PROMPT_VERSION = 'balanced-figure-v10.79-canvas-gravity'
MY3D_PAIR_PROMPT_VERSION = 'balanced-figure-v10.112-canvas-gravity'

# Shared human source interpretation only; this does not transform uploaded pixels.
# Bind original source identities before interpreting roll or planning output poses.
MATERIAL_COLOR_POLICY_TEMPLATE = 'MATERIAL COLOR — NEUTRAL LIGHT TAKES PRIORITY: Recover intrinsic material colors BEFORE applying identity, clothing or contact color locks. All color/hue/lightness/saturation preservation means those recovered colors, not tinted source pixels; removing illumination is not forbidden recoloring. Apply to Image 1 AND all identity anchors, {surfaces}. LIGHT VERSUS PIGMENT: Identify global or local illumination color, color-temperature shifts and reflected light across unrelated surfaces, including faint tint in shadows, sheen and rims. Matching tint in Image 1 and its identity crops repeats the same lighting evidence, not independent proof of {pigments}. {color_evidence} require coherent same-material regions or dye/print boundaries distinguishable from that lighting pattern; preserve them even when their genuine hue resembles the cast. {ambiguous_surface} proves neither a cast nor white material. MATERIAL RECOVERY: Where source evidence supports {neutral_materials}, retain that neutral family, value and texture; render shading/highlights in neutral grays without source-light tint. {natural_colors} If heavy/uniform lighting hides the original hue, infer conservatively without claiming exact recovery: no neutral patch does not permit retaining a recognized cast or forcing unsupported materials to white/gray/black. RELIGHT: Source lighting may vary {subjects}; infer each material locally, then {lit_subject} with one broad soft neutral-white frontal key and gentle fill. Remove source window/door/furniture projections, patchy sunlight and harsh side/backlight exposure patterns; recompute soft self/form/contact shadows from supported geometry, {shadow_geometry}. Preserve natural occlusion and 3D depth; {occlusion}. No {color_errors}global desaturation, darkened eyes, murky faces or washed-out surfaces. FINAL CHECK: Audit {audit_subject} under MATERIAL RECOVERY, especially dim regions and sheen. A white background alone is insufficient; remove residual source-light tint without changing genuine pigment, identity, ownership, construction, details or gentle self-shadow.'

PET_MATERIAL_COLOR_POLICY = MATERIAL_COLOR_POLICY_TEMPLATE.format(
    surfaces="the pet's fur, exposed skin, eyes, nose and owned collar, harness, clothing and accessories",
    pigments='fur, skin, eye or nose pigment or accessory dye',
    color_evidence='Natural coat colors, genuine fur markings, colored eyes/nose and accessory dyes or prints',
    ambiguous_surface='One colored surface or pale fur patch',
    neutral_materials='white/gray/black fur or accessories',
    natural_colors='Keep supported coat colors, patch boundaries and exposed skin/nose/eye pigmentation; do not turn colored fur white/gray, darken all coats or invent new markings from illumination.',
    subjects='between body regions and materials',
    lit_subject='light the complete pet',
    shadow_geometry='including muzzle and ears, fur masses, limbs, accessory folds and floor contact',
    occlusion='never move fur, ears, limbs or accessories or change pose to expose hidden areas',
    color_errors='bleached fur, uniform artificial coat tint, erased patches, ',
    audit_subject='every pet surface/material',
)

MATERIAL_COLOR_POLICY = MATERIAL_COLOR_POLICY_TEMPLATE.format(
    surfaces="every person's skin, hair, garment layers, shoes, accessories and belongings",
    pigments='complexion, hair dye or fabric pigment',
    color_evidence='Real dye, colored clothes, makeup, prints and markings',
    ambiguous_surface='One colored surface, skin alone or pale fabric',
    neutral_materials='white/gray/black clothes, bags or shoes',
    natural_colors="Keep each supported natural or dyed hair color; remove the recognized lighting cast from sheen/shadows without defaulting all hair to black or inventing a new intrinsic hair color from the illumination. SKIN EXPOSURE RECOVERY: Separate capture brightness from intrinsic complexion. Infer each person's depth/undertone from their own usable, unclipped skin regions across the face, neck and exposed limbs, allowing natural regional differences. Blown highlights and crushed shadows are missing color evidence, never proof of very pale or very dark skin. Render the inferred pigmentation with balanced exposure and readable tonal gradation. Correcting capture exposure comes before complexion/color locks and is not cosmetic whitening or tanning. If usable skin evidence is absent, infer conservatively without claiming exact recovery or using another person's complexion, hair, clothes or background as a skin-tone template.",
    subjects='between people and regions',
    lit_subject='light everyone',
    shadow_geometry='including facial features, hair/hands over skin, clothing folds and floor contact',
    occlusion='never move hair/hands or change pose to expose hidden areas',
    color_errors='whitening/tanning, a uniform artificial skin tint, identical skin brightness, ',
    audit_subject='every person/material',
)

SOURCE_ORIENTATION_OWNERSHIP_POLICY = "SOURCE IDENTITY AND OWNERSHIP: Read ACTUAL Image 1 for each accepted person's identity, body, clothes, belongings and actions/contact. Supplied face coordinates refer to ACTUAL Image 1 as transmitted; bind each accepted identity/anchor and A/B label to its corresponding face, never reorder by screen position. Trace each face-neck-shoulders-torso and owned shoulder-elbow-wrist-hand / hip-knee-ankle-foot chain through supported local occlusion, within the same continuous person and panel. Map garments, legwear/shoes and items/grips to that owner. Never bridge mirrors/reflections, doorways/frames, panels/screens, foreground overlap or background gaps. Proximity, vertical alignment, overlap and similar colors never transfer ownership; omit uncertain ownership rather than borrow details."



PAIR_RECONSTRUCTION_TARGET_POLICY = "PAIR RECONSTRUCTION TARGET: Rebuild exactly the two face-anchored people as complete 3D cartoon figures. Always render both upright and standing, regardless of source posture, with vertical necks and torsos, level shoulders and pelvis, straight neutral spines, heads centered above necks, and natural leg alignment. Complete unseen regions as standing anatomy. For a source-separated pair, choose contact after preserving each existing hand action; use a eligible source-unposed arm or direct body contact; keep both heads level, torsos straight and legs in a natural standing stance. Preserve clear gaze, facial direction and held objects. Head roll is exactly 0 degrees unless unmistakable same-person facial-midline evidence differs from both the neck axis and continuous torso axis after shared camera rotation is removed; if uncertain, keep 0 degrees."



PAIR_FINAL_POSE_AUDIT = 'FINAL PAIR POSE AUDIT — APPLY LAST: If converting a reclining or diagonally framed source person to standing, discard source head-to-image roll and align the head to the reconstructed vertical neck. A tilted source face or identity crop alone is never proof of local head tilt; only distinct same-person head-versus-neck-versus-continuous-torso axes can justify nonzero roll. Remove shared camera/photo roll. Render both people standing upright, with vertical necks and torsos, level shoulders and pelvis, straight neutral spines, heads centered above their necks, crowns directly above chins, noses centered over necks, eye lines horizontal, both ears level, and natural leg alignment, regardless of source pose. Do not cock, cant, slant or tilt a head toward the other person; head yaw/gaze may turn while roll remains zero. Every final person must stand upright in a natural neutral stance, regardless of whether the source shows sitting, crouching, reclining or lying. Keep both torsos vertical and straight and both heads level; never transfer global camera roll into the body or head. Fill cropped parts consistently with standing anatomy. For a separated pair, preserve supported arm/hand poses and connect only by moving the intact figures closer for natural adjacent body-surface contact; keep torsos straight, heads level and legs in natural standing alignment. Head roll remains exactly 0 degrees unless unmistakable same-person head-versus-neck-versus-continuous-torso evidence proves local tilt; if uncertain, keep 0 degrees. Do not copy a globally rotated photo as a diagonal figure; remove camera roll and retain natural upright standing alignment.'

PAIR_FINAL_STYLE_POLICY = "FINAL PAIR STYLE CHECK: Exactly two distinct, complete, non-photorealistic sculpted cartoon figures in one continuous portrait 2:3 composition on a clean solid white seamless background. Remove all source scenery, walls, doors, signs, furniture, objects and environmental detail; retain only the two people and soft contact shadows. Show each person head-to-feet in a natural upright standing pose. Keep a straight neutral torso, level head and naturally aligned legs; do not add a fashion lean, torso bend or head tilt without clear same-person evidence. Keep proportions along each person’s body axis, use one shared support/floor plane appropriate to that posture, and add soft contact shadows without any display base. Keep identities, clothes, objects and limbs separately owned."

PAIR_HUMAN_OUTPUT_TEMPLATE_POLICY = 'FINAL PAIR OUTPUT TEMPLATE — AFTER IDENTITY ANCHORS: Verify PAIR CAST PRIORITY, COMPOSITION and HEAD AND TORSO ORIENTATION. Apply the four-arm/contact check below.'

NO_ADDED_DISPLAY_BASE_RULE = "NO ADDED DISPLAY BASE — ABSOLUTE: Weight-bearing feet/paws or a resting body touch the studio floor under the composition rule. No separate disk, slab, stand, plinth, pedestal, platform, mounting plate or support post in any color or transparency."

HUMAN_NO_ADDED_DISPLAY_BASE_RULE = NO_ADDED_DISPLAY_BASE_RULE + " This bans presentation supports only: preserve genuine owned footwear, including platform shoes and their sole thickness, tread, shape and material color; do not flatten or remove soles."

FACE_PANEL_POLICY = """COLLAGE OVERRIDE: Use ONLY the clearest face-bearing panel for face, body pose, clothes and held items. Ignore all other panels, including back views of the same person. Never merge views; absent details are unseen."""

FIGURE_PROPORTION_POLICY = "OUTPUT HEAD SCALE — PLAN THE COMPLETE FIGURE FIRST: For EACH person, H is highest crown hair to chin, including the complete head but excluding hair hanging below the chin. F is crown to complete shoe soles, excluding canvas margins. Plan F as 100 units: H must be 24–25, aiming at 24.5; the chin-to-sole body occupies the remaining about 75.5 units. Establish this complete silhouette BEFORE detailed face rendering; transfer likeness into the planned head size. Apply the same target to head-only, partial-head, close, half-body and full-body references. Source/anchor zoom supplies facial relationships, NEVER output head size; do not extend a magnified avatar into a tiny body or shrink the body to accommodate the reference face. Keep a coherent neck, shoulders, torso, pelvis, limbs and shoes for that person's age and build; an adult retains adult cartoon anatomy, never toddler anatomy or elongated fashion-model legs. Preserve facial width-to-height relationships and the complete supported hairstyle. If the head is too large OR too small, uniformly rescale the whole head/crown and rebalance the complete figure; do not narrow the face, trim the hair or exaggerate shoulder width to hide an oversized head. Never shrink individual features, stretch legs, crop shoes or alter margins to simulate compliance. In a pair, use each person's OWN H and F, never the combined group bounds or an average; preserve supported relative stature without forcing equal head sizes or heights."

FIGURE_PROPORTION_FINAL_CHECK = "FINAL PROPORTION CHECK — EACH COMPLETE FIGURE: Assess every whole head-to-shoes silhouette under OUTPUT HEAD SCALE. If its ratio or visible head/body balance fails, correct the complete figure under that rule before finishing facial detail. Keep identity, age/build, owned details and supported poses/contacts. No rulers, guides, measurement text or extra panels."

POSE_AXIS_PROOF_GATE = "HEAD AND TORSO ORIENTATION — SOURCE-RELATIVE EVIDENCE FOR EACH PERSON: Separate common image rotation from anatomical bends BEFORE normalizing posture. Image 1 may already have been rigidly rotated to level a face; neither a level eye line in that canvas nor a preprocessing angle proves an upright body or a neutral head. Compare the same connected head, neck, both shoulder roots and visible upper torso; their relative geometry survives common image rotation. Use coherent scene/support and body evidence to separate shared roll, never the eye line alone. HEAD ROLL — TWO EXCLUSIVE CASES: PROVEN LOCAL BEND: Reliable facial-midline, neck attachment and shoulder/upper-torso evidence jointly showing a lateral head-versus-torso bend authorizes retaining that bend, its side and supported degree, in the reconstructed standing frame. The neck may curve with the head: it need not be a third distinct straight axis. Cropped waist/legs do not invalidate sufficient upper-body evidence. Do not independently straighten this evidenced head while correcting common rotation or completing the body. NO PROVEN LOCAL BEND: If those relationships are missing, ambiguous or show only common rotation, render head roll exactly 0 degrees in the upright output frame, centered over its own neck; never guess a decorative tilt. A slanted face, eye line, hair, one shoulder, crop border or identity headshot alone is insufficient. Yaw and pitch are governed separately by FACE DIRECTION; they cannot establish lateral roll. Never lean the body to compensate for head orientation. OUTPUT BODY FRAME: Whole-person rigid roll is 0 degrees relative to CANVAS VERTICAL, with gravity down and a horizontal floor. Keep torso/pelvis mass supported over the weight-bearing footprint; level eyes or soles alone do not prove an upright figure. Remove the shared diagonal component across trunk/pelvis/supporting legs from the complete figure, preserving internal joints and ARM POSE AND HAND GEOMETRY; never counter-tilt only the head or merely shift shoes. LOCAL TORSO: Default to a straight neutral spine with naturally stacked shoulders/ribcage/waist/pelvis. Permit a curve only when continuous same-person shoulder/ribcage/waist/pelvis evidence proves local bending after de-roll; preserve its side and degree without forcing chest/waist/pelvis centers collinear. Missing waist/pelvis, folds, limb/head pose, one shoulder, perspective, borders or photo roll alone prove no bend. No invented arch, chest thrust, hip-pop, S-curve or fashion lean. A proven local curve cannot authorize rotating the whole figure. Apply per person, never cancel opposing leans across a pair."

FINAL_POSE_AXIS_AUDIT = 'FINAL POSE-AXIS AUDIT: Recheck HEAD AND TORSO ORIENTATION against Image 1, including its full three-axis head-roll exception, independent yaw/pitch and proven local torso curves. Restore the upright gravity frame; remove unsupported lean, arch or hip-pop.'

CLOSE_PORTRAIT_CLOTHING_EVIDENCE_GATE = "SOURCE CLOTHING LOCK — EVERY FRAMING: Before design candidates or scene removal, classify each owner's upper clothing, lower clothing and footwear as observed, partly observed or wholly absent in Image 1. Even small observed fragments lock category, layers/order, coverage, construction, fabric and intrinsic color; unclear category or fabric subtype does not erase other known attributes. Treat a removed occluder's footprint as a gap in the SAME garment, not a wholly absent region: join only that owner's visible garment fragments through the gap, continuing their silhouette, coverage and surface, never the occluder's shape/material. Where details are unresolved, use minimal plain continuation compatible with the evidence, not a familiar redesigned garment. Missing hems/sleeve ends, cropping, shadow or low resolution never unlock observed clothing. Candidate styles/palettes apply ONLY to their corresponding wholly absent region; they cannot replace or coordinate away source clothes. Visible upperwear stays separate unless continuous source construction across the waist proves a one-piece; otherwise convert a one-piece candidate to a separate lower completion. A proven one-piece retains its supported upper construction. Preserve owned neck jewelry separately; never transfer clothing evidence from another person."

UPPER_LAYER_EVIDENCE_GATE = 'UPPER LAYER GATE — EACH PERSON: Under SOURCE CLOTHING LOCK, outerwear requires positive same-person Image 1 evidence; retain its order, never add an over-layer. A single top continues its own fabric through missing or hair-obscured shoulder/chest/arm areas. Keep neckline shape and anatomical landmarks: shoulder endpoints, shoulder-relative width, collarbone/sternum-relative depth and exposed neck/chest/shoulder skin. Do not raise/narrow/deepen/widen that opening or add an unsupported collar/band. Scale the construction with the stylized torso. Only for wholly absent upperwear may one simple separate top be designed, without outerwear.'
FINAL_UPPER_LAYER_AUDIT = 'FINAL UPPER LAYER AUDIT — AFTER WARDROBE/ANCHORS: Verify SOURCE CLOTHING LOCK and UPPER LAYER GATE against each owner; discard candidate details that displaced source evidence.'

FINAL_CLOTHING_EVIDENCE_AUDIT = "FINAL CLOTHING AUDIT: Enforce CLOSE-PORTRAIT CLOTHING LOCK for each owner; a cropped top stays separate unless continuous waist-crossing construction proves a one-piece. Repair mismatches without redesigning visible evidence."

BODY_CONTOUR_POLICY = "BODY CONTOUR: Preserve genuine source-supported anatomy, body proportions and maturity, never stereotypes inferred from hair or clothing. With unclear torso evidence, use a modest, natural, unaccentuated body contour; do not invent exaggerated musculature, chest volume or body shape from fabric folds. Preserve any clearly source-supported body contour without flattening or reinterpreting it. Invented stance: relaxed shoulders, ribs over pelvis; no thrust-forward chest or arched back."

CLOSE_PORTRAIT_SCALE_POLICY = "FACE-ONLY REFERENCE: Do not extend the screenshot downward with a tiny body. Discard close-lens head enlargement; retain identity, expression, hair, source-supported facing direction and visible details. Never turn an adult headshot into toddler anatomy."

CLOSE_PORTRAIT_HEAD_ROLL_POLICY = "HEADSHOT DEFAULT: A headshot's diagonal face and cropped neckline are not anatomical tilt proof. Render the neutral upright head required by HEAD ORIENTATION; only the complete three-axis comparison in Image 1 may authorize a genuine local tilt. Keep independently proven yaw and pitch without lateral roll."

HUMAN_ONLY_POLICY = "SOURCE SUBJECT FILTER — PEOPLE AND POSSESSIONS: Classify Image 1 entities by SOURCE construction and context before stylization. Worn/carried representations with evidence of their own manufactured construction (stitching, stuffing or molding) are inanimate possessions; their faces/limbs do not count as extra people, animals or human anatomy. Animal-like shape alone does not prove life; the output's sculpted finish cannot reclassify a source animal. Exclude source-depicted living animals regardless of ownership/contact. Scene structures/supports and environmental occluders stay excluded at any depth despite touch/grip/lean/overlap. Independent carried items remain eligible despite partial occlusion or ground contact; omit uncertain eligibility. This classification overrides object/grip/contact preservation. Remove excluded entities without remnants/substitutes; restore anatomy and apply SOURCE CLOTHING LOCK to garment gaps. Release only dependent contact at the supported arm/wrist position under ARM POSE AND HAND GEOMETRY."

PAIR_HUMAN_ONLY_POLICY = HUMAN_ONLY_POLICY
PAIR_POSE_PRESERVATION_POLICY = POSE_AXIS_PROOF_GATE

AMBIGUOUS_HAND_POLICY = "AMBIGUOUS-HAND SAFETY OVERRIDE — HIGHEST PRIORITY: First decide separately for each hand whether its wrist, palm and individual finger bases are clearly resolved. Motion blur, compression, a tiny hand, a partial silhouette or indistinct fingers do NOT prove a gesture, pointing pose, digit count, grip, palm direction or object. They do NOT erase a clearly supported shoulder, elbow, forearm, wrist position or hand height. Before rebuilding an unclear hand, lock its shoulder → elbow → wrist chain and its vertical band relative to the torso: a source wrist/hand held in front of the chest must remain in front of the chest; a hand visibly above the upper waist must never drift to the waist, hip or side. If a hand is not clearly resolved, preserve that locked chain and location, then rebuild only the hand at that same wrist as one SIMPLE RELAXED CLOSED-HAND SILHOUETTE: palm obliquely toward the torso, fingers gently folded together into one smooth natural knuckle arc, thumb resting naturally at the side or lightly across the folded fingers. This is a complete five-digit hand with normal natural occlusion, not a mitten, stump or hidden arm end; do not visually separate or emphasize every fingertip. Never imitate blur, guess individual finger directions, point, spread fingers toward the camera, make a sign or force a grip. Do not lower, straighten, hide or reposition a clearly raised arm merely because its fingers are blurred. A clear held item and its clear grip remain source evidence, but an unclear hand never authorizes malformed anatomy."

FINAL_AMBIGUOUS_HAND_POLICY = "FINAL HAND-FINGER OVERRIDE — APPLY LAST: Before returning, inspect every source hand one last time. Unless its palm and individual finger bases are clearly visible, it is NOT evidence for a pointing, counting, peace-sign, pinching, waving or splayed-finger pose. Preserve every reliable shoulder → elbow → forearm → wrist chain and its original vertical band relative to the torso, including a clearly raised arm and its hand height: a chest-level source hand must still be chest-level, and a hand above the upper waist must never end at the waist, hip or side. Replace only the unclear finger gesture at that same wrist with one SIMPLE RELAXED CLOSED-HAND SILHOUETTE: a complete continuous-wrist hand, palm obliquely toward the torso, four folded fingers forming one smooth knuckle arc and a naturally placed thumb, with no invented object contact. Natural occlusion is required; do not force a front-facing open palm, individually separated fingertips or a display gesture. This rule overrides guessed finger gestures but never overrides a clearly supported arm pose, height or direction. Never return a partly guessed finger gesture, and never lower or straighten a raised arm just to simplify anatomy."


def build_my3d_pet_prompt() -> str:
    """Create the isolated GPT-image prompt for one validated cat or dog."""
    return "\n".join([
        "Reconstruct exactly one validated real cat or dog in Image 1; no human, toy animal, second pet, hybrid or unaccepted reflected/background subject.",
        "PRESERVE PET IDENTITY: Keep species, visible breed-like features, intrinsic fur color/markings/length/texture, ear shape, eye color, muzzle, nose color, build, tail, expression and clearly visible collar/tag/harness/clothes. Preserve ear position relative to the corrected head. Invent no accessories or markings.",
        SCULPTED_STYLE_POLICY + " PET SURFACES: Use sculpted fur masses rather than photoreal individual fur strands.",
        "ANATOMY: One head, two ears, four legs and four paws and one tail when source-supported. No human anatomy or invented human clothing, extra limbs, fused paws, duplicate tails or distorted features. Complete hidden anatomy naturally.",
        "UPRIGHT PET — ABSOLUTE: Correct body AND head to right-side up regardless of source/anchor tilt, rotation, inversion, holding or leaning. Head naturally above neck, ears up and muzzle/chin below eyes; preserve expression and supported face direction without roll. No sideways/inverted head or diagonal whole-body lean. Allow standing, sitting, walking or belly-down lying only under normal gravity on a horizontal floor; a lying torso may be horizontal, never rolled onto its side/back. Adapt an unsupported, held, tilted or inverted pose to balanced natural sitting/standing while retaining visible anatomy, markings and accessories; add no human support.",
        NO_ADDED_DISPLAY_BASE_RULE,
        *_my3d_output_blocks(pet=True),
    ])

HEAD_EVIDENCE_POLICY = "HEAD ORIENTATION — NORMALIZE BEFORE RENDERING: Start with a neutral, upright head balanced vertically on the neck. Repose the source face into that neutral roll while retaining identity, facial relationships, expression, hair and independently proven yaw/pitch. Never preserve the face's diagonal angle in either Image 1 or a later identity crop. Identity crops describe appearance, not head roll. A nonzero anatomical roll is allowed ONLY if Image 1 clearly resolves all three comparable axes: (1) head facial midline, (2) the same person's neck centerline, and (3) the same person's continuous torso/sternum centerline with both shoulder-to-neck connections. Remove their COMMON camera/photo rotation first, then require an unambiguous head-versus-neck-and-torso bend. A neck fragment, neckline, one shoulder, angled headshot, partial upper chest, sloping pupil line or a diagonally framed person cannot establish this comparison. If any axis/comparison is missing, unclear or aligned, USE THE NEUTRAL UPRIGHT HEAD. For that default, the nose bridge, philtrum and chin share a vertical facial midline aligned above the neck; the pupil centers are level for a frontal view. Preserve local facial asymmetry without slanting that midline. Do not balance a tilted face by tilting the torso. Do not invent a fashion lean or cute head tilt."

FACE_DIRECTION_POLICY = 'FACE DIRECTION — YAW/PITCH ONLY: Use yaw 0 and neutral forward gaze unless multiple source landmarks jointly prove a turn: nose bridge/tip direction, near/far eyes and cheek/jaw contour. Preserve that proven side/profile direction. Crop edges, a cut-off or hair-hidden eye, asymmetry, gaze alone and missing face regions cannot prove yaw. Ambiguous yaw stays frontal, without an invented profile or three-quarter turn. Preserve proven up/down gaze and pitch independently. This gate does not reset head roll decided by HEAD AND TORSO ORIENTATION.'

SCULPTED_STYLE_POLICY = 'STYLE — SCULPTED CARTOON COLLECTIBLE: Use premium painted vinyl/resin with rounded sculpted forms, painted eyes and smooth matte surfaces. Model the face AND complete body in the same visibly non-photorealistic sculpture language; no live-subject render, photographic face patch or realistic face pasted on a toy.'

STUDIO_BACKGROUND_POLICY = 'STUDIO OUTPUT: One continuous portrait 2:3 image on a level seamless studio floor/background with actual fully opaque pure-white (#FFFFFF) background pixels and soft floor/contact shadows. No transparency, alpha cutout, checkerboard, empty/black/gray field or copied source environment. Remove scenery, UI, overlay text, watermarks and stray unowned fragments; retain accepted subjects and supported owned details. Return only the generated image.'

NON_PHOTOREALISTIC_FIGURINE_POLICY = SCULPTED_STYLE_POLICY + ' HUMAN SURFACES: Use simplified carved nose/lips, sculpted rounded facial planes, grouped solid hair locks and simplified garment folds. Keep source eye size/spacing and eyelid state, nose/mouth proportions, face shape, age, intrinsic complexion and expression/asymmetry recognizable; painted expression must not enlarge eyes or shrink nose/mouth. Preserve an adult as an adult. Hair is molded locks with a few carved grooves, not individually realistic strands. Clothes retain visible category/material color/construction as sculpted painted surfaces rather than real woven macrotexture. No photographic skin or fashion photography.'


FINAL_HEAD_AND_STYLE_POLICY = "FINAL TARGET-FRAME CHECK: Recheck the whole-person standing gravity frame and retained source torso/joint relationships using RECONSTRUCTION TARGET after common camera-roll removal. The face midline is upright on the neck unless the original three-axis comparison proves a local relative bend. Retain source-supported local torso curves and shoulder–waist–pelvis relationships without forcing their centers onto one line; remove only the complete person's common camera/photo rotation. Render the whole figure as sculpted painted resin with the measured four-unit silhouette."

BODY_UPRIGHT_POLICY = "WHOLE-FIGURE ORIENTATION — NORMALIZE BEFORE RENDERING: Use an upright standing gravity frame with a horizontal floor. Remove shared camera/photo roll from the COMPLETE person, including head, neck, trunk, pelvis and limbs, without rotating the whole output figure diagonally. Preserve the person's source-supported local torso/spinal curves, shoulder–waist–pelvis relationships, gestures and joint bends after removing that common rotation. A genuinely bent/slanted trunk is allowed within the upright frame; do not force chest, waist and pelvis centers to be collinear or straighten source-supported curvature. A local bend must not become a common lean of the entire person. Missing body regions are completed naturally, without invented fashion offsets. Camera angle, folds or unrelated limb/head action alone cannot create a trunk bend. Local head tilt still requires the independent three-axis proof above. Both shoes rest naturally on a level floor."

RECONSTRUCTION_TARGET_POLICY = "RECONSTRUCTION TARGET: Build a new upright four-unit cartoon resin figure, then transfer the source person's likeness, hair and clothing into that figure's local coordinates. Remove whole-person camera/photo roll without straightening source-supported local torso curves or joint relationships. Establish a level studio floor, upright standing frame and neutral head balanced squarely on the neck. Preserve clear yaw, pitch, expression, gestures, held items and source-supported local torso bends. A curved/slanted trunk is allowed inside the upright frame; chest, waist and pelvis need not be collinear. Permit head roll ONLY when Image 1 clearly shows the same person's facial midline, neck centerline and continuous torso/sternum centerline with both neck-to-shoulder connections. Remove their common camera rotation first; preserve only an unambiguous remaining head-versus-neck-and-torso bend. Missing, unclear or aligned axes keep the neutral head. Later identity crops supply likeness only and have no authority over head/body orientation."

HUMAN_OUTPUT_TEMPLATE_POLICY = "FINAL OUTPUT TEMPLATE — AFTER IDENTITY ANCHORS: Verify SINGLE CAST AND ANATOMY, COMPOSITION and HEAD AND TORSO ORIENTATION; correct mismatches under their source-evidence rules."

def _build_close_portrait_prompt(wardrobe_profile, wardrobe_palette, footwear_profile):
    """Use the same complete single-person policy for close references."""
    return build_my3d_direct_prompt(
        wardrobe_profile, wardrobe_palette, footwear_profile=footwear_profile,
        source_context="close_portrait",
    )

# A single policy shared by every wardrobe/palette. This instructs the image
# model; it is not a local pose detector or a guarantee of generated geometry.
HEAD_POSE_POLICY = "HEAD ORIENTATION — NORMALIZE BEFORE RENDERING: Start with a neutral, upright head balanced vertically on the neck. Repose the source face into that neutral roll while retaining identity, facial relationships, expression, hair and independently proven yaw/pitch. Never preserve the face's diagonal angle in either Image 1 or a later identity crop. Identity crops describe appearance, not head roll. A nonzero anatomical roll is allowed ONLY if Image 1 clearly resolves all three comparable axes: (1) head facial midline, (2) the same person's neck centerline, and (3) the same person's continuous torso/sternum centerline with both shoulder-to-neck connections. Remove their COMMON camera/photo rotation first, then require an unambiguous head-versus-neck-and-torso bend. A neck fragment, neckline, one shoulder, angled headshot, partial upper chest, sloping pupil line or a diagonally framed person cannot establish this comparison. If any axis/comparison is missing, unclear or aligned, USE THE NEUTRAL UPRIGHT HEAD. For that default, the nose bridge, philtrum and chin share a vertical facial midline aligned above the neck; the pupil centers are level for a frontal view. Preserve local facial asymmetry without slanting that midline. Do not balance a tilted face by tilting the torso. Do not invent a fashion lean or cute head tilt."

COMPLETE_SUBJECT_FRAMING_TEMPLATE = 'COMPOSITION — COMPLETE SUBJECT BOUNDS (FINAL FRAMING): Finish {pose_completion} BEFORE choosing camera distance. On normalized canvas coordinates x=0–100 and y=0–100 from top to bottom, fit the ENTIRE {subject_bounds} AND required owned items within x=10–90 and y=10–90. Measure the outermost silhouette, including {extremities}; the highest actual extremity sets the top, {upper_bound}. Any edge contact or cut-off extremity fails this framing check: pull the camera back and recenter the {recenter_subject} until it fits, preserving {preserved}. Keep genuine source-supported occlusion. After fitting, do not zoom in or crop to fill the canvas; render no coordinate lines, border or labels.'

PET_FULL_FRAME_COMPOSITION_POLICY = COMPLETE_SUBJECT_FRAMING_TEMPLATE.format(
    pose_completion='the supported pet pose and missing anatomy under UPRIGHT PET and ANATOMY',
    subject_bounds='pet',
    extremities='ear tips, fur outline, all paws, the source-supported tail and accessory ends',
    upper_bound='not merely the skull when ears or fur extend higher',
    recenter_subject='complete pet',
    preserved='species-appropriate anatomy, build, the permitted pet pose and genuine accessory contacts',
)

HUMAN_FULL_FRAME_COMPOSITION_POLICY = COMPLETE_SUBJECT_FRAMING_TEMPLATE.format(
    pose_completion='the supported pose and source-cropped limb/hand continuations',
    subject_bounds='accepted cast',
    extremities='raised fingertips, side-reaching elbows/hands, hair, shoes and item ends',
    upper_bound='NOT the crown when a hand is higher',
    recenter_subject='accepted cast',
    preserved='OUTPUT HEAD SCALE, ARM POSE AND HAND GEOMETRY, relative stature and eligible contacts',
)

HUMAN_POSE_LIMBS_POLICY = "POSE/LIMBS: Always render standing on the person's own feet, including seated/crouched/riding/reclining sources; keep clear local gestures/steps/item contacts and supported limb joint bends after de-roll, never mirror. Apply ARM POSE AND HAND GEOMETRY to both arms. Trace each owned hip→thigh→knee→lower leg→ankle→foot. Complete unclear limb segments without ending limbs at clothes/bags/body edges or leaving floating hands/feet."

HUMAN_OWNED_DETAILS_POLICY = "OWNED DETAILS: Before posing, inventory each person's possessions admitted by SOURCE SUBJECT FILTER; every admitted item is required output content, including simultaneous/partly hidden items, headwear/hair ornaments, eyewear, jewelry/watch and bags/straps/charms. Retain tattoos/body marks. Preserve source category/count/intrinsic color/material/pattern/scale/side/orientation/landmark/attachment/contact. Reproduce an identified owned object's supported form even if its subtype or tiny details are unclear; omit only unproven objects/ownership. Isolated hair/seams/shadows/highlights/faint lines prove no accessory. Do not invent objects or cover/relocate readable tattoos."

HUMAN_GARMENT_DETAILS_POLICY = 'GARMENT DETAILS: Within the clothing gates, retain fit, seams/pockets/waist/fastenings, hem, visible markings, texture and material color even when category is unclear. Preserve socks/tights/hosiery/bare-leg presence, absence, coverage and shoe connection. An unseen hem alone does not prove long trousers.'

SINGLE_CAST_PRIORITY_GATE = "SINGLE CAST AND ANATOMY: Reconstruct exactly ONE complete person, identified by the validated face in Image 2 and its match in Image 1: one head/face/torso and two complete arms/hands/legs/feet. Incidental faces/body fragments inside either reference do not add identities. Exclude all other people and their parts even when they overlap, touch or interact with the target; those relationships cannot authorize a partner or provide target anatomy/clothes. Cast exclusion overrides interaction/grip/occlusion preservation; release only contact dependent on the excluded person, then reconstruct the target's missing anatomy/clothing under ARM POSE AND HAND GEOMETRY and SOURCE CLOTHING LOCK."

SINGLE_PERSON_PROMPT_TEMPLATE = "RECONSTRUCTION TARGET: Plan the complete figure under OUTPUT HEAD SCALE before transferring its accepted identity and owned details.\n\n{face_panel}\n\n{figurine_style}\n\nIDENTITY: Preserve source hair length/style, expression and asymmetry under the style and identity-bound completion rules. Do not infer identity from clothes/hair stereotypes or substitute a stock face, invented makeup/facial hair or sexualization. If identity is unreliable, use neutral adult presentation.\n\n{face_direction}\n\nUNSEEN-REGION CANDIDATES — INACTIVE UNTIL SOURCE CLOTHING LOCK PERMITS THE MATCHING REGION: lower/outfit candidate {wardrobe}; upper candidate {upper}; intrinsic palette candidate {palette}; color suggestions {colors}; footwear candidate {footwear}. Within permitted unseen areas, use varied harmonious age-appropriate colors/fabrics/patterns without routinely defaulting to plain long trousers, all-black/all-white outfits or white sneakers. Optional hosiery requires lower clothes, legwear AND shoes all unseen, clearly adult feminine presentation and no conflicting evidence or uncertain age.\n\n{composition}\n\n{display_base}"

LOWER_BODY_SOURCE_EVIDENCE_GATE = "LOWER-BODY EVIDENCE — BEFORE DESIGN: Inspect each owned waist-to-feet chain and bottom/side image edges. Lock every small, cropped, occluded or shadowed lower-garment/shoe fragment's category, construction, material color, fabric, hem/leg openings, bare-skin/legwear coverage and markings. Shorts leg openings with bare thigh prove shorts and short coverage on both legs, never a skirt/dress/long trousers; proven skirts/trousers likewise retain category. Invent no denim, stitching, pleats, buttons, belts, pockets or prints. Sitting-to-standing changes joints, not garment category/hem/coverage/fabric/material colors; separate shorts legs without merging into a skirt.\nPARTIAL SHOE IS NOT FREE FOOTWEAR: ANY owned shoe fragment excludes BOTH shoes from random footwear/palette/accent design. Extend that same pair's visible intrinsic color/material/pattern/closure/foot coverage, without unsupported colored panels/soles. A cropped toe/sole is no permission for a new shoe family, heels/bows/open sandals. An uninterrupted vamp/strap with markings remains that broad top-facing panel: no invented lace rows/eyelets or migration of marks onto sneaker sidewalls. Do not infer a popular brand from light color or an angular mark. Preserve readable mark placement/scale/stroke geometry/tonal layout; substitute no logo/letter/stripe/emblem. Unreadable characters stay muted source-shaped marks, never invented readable branding. Add no laces/straps/bows/heels/perforations/contrast panels without source construction evidence. Missing toe/heel/sole gets minimal compatible continuation. One readable shoe completes the other as the SAME pair unless clear evidence proves mismatch, never a laced/slip-on hybrid. These are conditional source locks, not default styles. Complete only unobserved attributes; never borrow from another person or background."
FINAL_LOWER_BODY_FIDELITY_AUDIT = "FINAL LOWER-BODY FIDELITY AUDIT: Verify LOWER-BODY EVIDENCE and PARTIAL SHOE IS NOT FREE FOOTWEAR against each owner's source; correct only mismatches."

IDENTITY_BOUND_COMPLETION_GATE = "IDENTITY-BOUND BODY/UNSEEN OUTFIT — BEFORE DESIGN: Lock each person's own anchored identity, age and gender presentation before choosing body or clothes. Never change a man/boy into a woman/girl or vice versa. Hair length, crop, screenshot text, another person and random design candidates cannot decide gender/anatomy. For an unseen masculine torso use a natural male chest with a flat garment front, no invented breasts/bust shaping, cleavage or hourglass hips. Uncertain presentation uses a modest neutral contour. Preserve genuine visible contours. For wholly unseen masculine/uncertain clothing choose separate trousers, jeans or tailored shorts, one simple separate T-shirt/shirt/knit/sweatshirt if the top is unseen, and flat neutral sneakers/canvas shoes/loafers/casual lace-up shoes if both shoes are unseen. Ignore random skirts/dresses/bodysuits/feminine blouses, ballet flats/Mary Janes, bows, ankle straps and heels. Colors/patterns may vary within these categories. Clearly feminine people retain age-appropriate completion freedom, without exaggerated anatomy. Visible source clothes/shoes/contours ALWAYS win regardless of gender, including a man actually wearing a skirt or heels. Defaults affect missing evidence only. Apply independently per person; never borrow anatomy or clothing."

FINAL_IDENTITY_COMPLETION_AUDIT = "FINAL IDENTITY COMPLETION AUDIT — AFTER ANCHORS: Verify each owner's completion under IDENTITY-BOUND BODY/UNSEEN OUTFIT."

OUTPUT_CLARITY_POLICY_TEMPLATE = 'OUTPUT CLARITY — {scope}: Render {render_subject}. Source photos and anchors provide identity and supported details, never focus quality or photographic patches. Do not copy blur, low resolution, noise, compression grids, posterization, isolated blocky dark/color patches, crop edges or exposure seams. An unclear pixel patch alone proves no {uncertain_features}. Compare the complete source with its matching anchor; reconstruct degraded regions as coherent sculpted surfaces and soft studio shading across {surface_regions}, without rectangular patches or pasted-face joins. Preserve supported {retained_features} and plausible self-shadow; {no_whitening} or erasure of reliable features. Resolve {details} recognizable sculpted cartoon form, never a generic sharp face. Keep expressions, eye closure, natural occlusion, ownership and unclear-detail rules; invent no hidden text, markings, {invented_detail} for sharpness. Smooth matte resin is not defocus: no motion/focus/depth-of-field blur, haze, smearing, pixelation, sharpening halos or gritty texture. FINAL CHECK: repair local blur and unsupported blocks/seams on every face and body while preserving evidenced identity, material colors and natural shading.'

PET_OUTPUT_CLARITY_POLICY = OUTPUT_CLARITY_POLICY_TEMPLATE.format(
    scope='THE COMPLETE PET',
    render_subject='the complete pet tack-sharp, equally resolving its face, body and accessories at any depth',
    uncertain_features='new fur spot, scar, fur shape or pigment boundary',
    surface_regions='forehead, ears, muzzle, jaw and fur',
    retained_features='coat colors, genuine markings, scars, ear/fur outlines and occlusion',
    no_whitening='no coat bleaching, identity smoothing',
    details="eyelids, source eye shapes and visible irises/pupils, ears, muzzle, nose, mouth, fur masses, legs, paws, tail and accessories in that pet's",
    invented_detail='objects or paw digits',
)

HUMAN_OUTPUT_CLARITY_POLICY = OUTPUT_CLARITY_POLICY_TEMPLATE.format(
    scope='ALL PEOPLE, FACE AND BODY',
    render_subject='each complete figure tack-sharp, equally resolving both faces and bodies at any depth',
    uncertain_features='mole, birthmark, makeup, scar, hair shape or skin-color boundary',
    surface_regions='forehead, cheeks, jaw, hairline and hair',
    retained_features='complexion, genuine markings, makeup, scars, hairline/occlusion',
    no_whitening='no whitening, identity smoothing',
    details="eyelids, source eye shapes and visible irises/pupils, brows, nose, lips, hair masses, hands, clothing edges and footwear in that person's",
    invented_detail='objects or fingers',
)


HUMAN_HELD_OBJECT_HAND_POLICY = "ARM POSE AND HAND GEOMETRY — HARD CONSTRAINTS FOR EACH PERSON: Apply equally to BOTH arms, empty hands, held items and supported contact. ARM LOCK: Before rendering, trace each owned shoulder→upper arm→elbow→forearm→wrist→palm. Preserve supported upper-arm direction, elbow position/bend, forearm direction, wrist height relative to the face/chest/waist, front/side/back placement and palm/grip orientation. Both raised arms stay raised; never lower, straighten, put behind the back or into pockets, or conceal a source-visible arm/hand behind hair, clothes, the body or a partner to simplify anatomy or composition. Standing conversion and head/body scaling map this configuration into the same person's upright body frame, never reset it to a generic pose. Keep each item's anatomical hand, count and grip/contact type; two hands on one item remain on it. LOCAL HAND EVIDENCE: Assess EACH source hand independently; overall photo, face or background blur does not trigger finger repair when that hand is readable. CLEAR HAND: Preserve its resolved finger arrangement, open/closed shape, palm facing and grip; do not replace it with a generic folded hand. DEGRADED HAND ONLY: When motion/defocus blur, low resolution or compression leaves digit boundaries unresolved, retain reliable palm facing, open/closed silhouette, finger-group direction and contacts at the locked wrist. Rebuild only unresolved finger geometry as a simple coherent anatomical hand; do not trace blur trails, duplicate edges or compression patches as separate digits or guess a complex sign. A supported held item or existing contact requires its compatible grasp, never the empty-hand fallback. ONLY an empty hand with no reliable gesture/open-closed silhouette may use gently curled relaxed fingers and a naturally resting thumb, aligned with the supported wrist/palm plane; invent no pointing, sign, pinch, wave, splayed display or face contact. OCCLUSION IS NOT BLUR: Keep genuine hidden fingers naturally occluded; do not expose them for clarity. A cropped wrist/hand or unclear fingers do not make a visible arm unknown: continue absent segments along its evidenced direction and joint bends, without inventing an action for the missing endpoint. Only an arm with NO reliable pose evidence may default to a relaxed empty hand at its side. HAND ANATOMY — NON-NEGOTIABLE: Every hand has one continuous wrist and palm, exactly four fingers rooted at distinct knuckles and one opposable thumb with its own base; use natural digit lengths, thickness, spacing and joint bends. No extra, missing, duplicated, fused, melted, branching, twisted, reversed-joint, intersecting or detached digits, second palm, mitten, stump or partial palm. Natural occlusion may hide digits; do not expose five fingertips or force all fingers into a grip. Repair only malformed local hand geometry while retaining the locked arm, wrist and supported gesture/contact; never hide/delete a hand, drop an item or move the arm as a repair. OBJECT SEPARATION: Keep skin and objects distinct. Trace rigid objects behind gripping fingers and flexible handles/straps/loops through the grip opening, never through flesh; supported fingers hook around the load-bearing section with the load naturally below contact. Strap edges/folds are not digits; no floating handle, fused skin/material or object-grown finger. FINAL ARM/HAND CHECK: Compare each arm to its source lock, then trace every visible digit to its own palm and every object through a physically possible grip. Correct local failures under these constraints before finishing; preserve identity, clothes, limb count and eligible mutual contact."


def _assemble_my3d_human_prompt(
    *,
    pose_blocks: tuple[str, ...],
    character_blocks: tuple[str, ...],
    cast_blocks: tuple[str, ...] = (),
) -> str:
    """Own the shared human rule order; callers supply only mode-specific sections.

    Keep the standalone body complete. The route moves final rules after identity
    bindings through append_my3d_human_prompt_audits, without repeating them.
    """
    return "\n\n".join((
        *cast_blocks,
        SOURCE_ORIENTATION_OWNERSHIP_POLICY,
        MATERIAL_COLOR_POLICY,
        *pose_blocks,
        FIGURE_PROPORTION_POLICY,
        CLOSE_PORTRAIT_CLOTHING_EVIDENCE_GATE,
        UPPER_LAYER_EVIDENCE_GATE,
        LOWER_BODY_SOURCE_EVIDENCE_GATE,
        IDENTITY_BOUND_COMPLETION_GATE,
        HUMAN_ONLY_POLICY,
        HUMAN_POSE_LIMBS_POLICY,
        HUMAN_OWNED_DETAILS_POLICY,
        HUMAN_GARMENT_DETAILS_POLICY,
        *character_blocks,
        STUDIO_BACKGROUND_POLICY,
        FINAL_UPPER_LAYER_AUDIT,
        FINAL_LOWER_BODY_FIDELITY_AUDIT,
        FINAL_IDENTITY_COMPLETION_AUDIT,
        HUMAN_HELD_OBJECT_HAND_POLICY,
        HUMAN_OUTPUT_CLARITY_POLICY,
    ))


def build_my3d_direct_prompt(
    wardrobe_profile: str | None = None,
    wardrobe_palette: str | None = None,
    *,
    upper_garment_profile: str | None = None,
    footwear_profile: str | None = None,
    source_context: str = "general",
) -> str:
    """Same complete policy for general/close; parameters affect unseen regions only."""
    wardrobe = WARDROBE_COMPLETIONS.get(wardrobe_profile or "", WARDROBE_COMPLETIONS["shorts"])
    palette = WARDROBE_PALETTES.get(wardrobe_palette or "", WARDROBE_PALETTES["pastel"])
    colors = UNSEEN_COLOR_DIRECTIONS.get(wardrobe_palette or "", UNSEEN_COLOR_DIRECTIONS["pastel"])
    upper = UPPER_GARMENT_COMPLETIONS.get(upper_garment_profile or "", "a coordinated upper-garment design")
    footwear = FOOTWEAR_STYLES.get(footwear_profile or "", FOOTWEAR_COMPLETIONS.get(wardrobe_profile or "", FOOTWEAR_COMPLETIONS["shorts"]))
    character = SINGLE_PERSON_PROMPT_TEMPLATE.format(
        wardrobe=wardrobe, palette=palette, colors=colors, upper=upper, footwear=footwear,
        face_panel=FACE_PANEL_POLICY, figurine_style=NON_PHOTOREALISTIC_FIGURINE_POLICY,
        face_direction=FACE_DIRECTION_POLICY,
        display_base=HUMAN_NO_ADDED_DISPLAY_BASE_RULE,
        composition=HUMAN_FULL_FRAME_COMPOSITION_POLICY,
    )
    return _assemble_my3d_human_prompt(
        cast_blocks=(SINGLE_CAST_PRIORITY_GATE,),
        pose_blocks=(POSE_AXIS_PROOF_GATE,),
        character_blocks=(character,),
    )


PAIR_ANCHOR_ARM_OWNERSHIP_AUDIT = "FINAL CAST/ARM/CONTACT AUDIT — AFTER ANCHORS: Under ARM POSE AND HAND GEOMETRY, verify A-left/A-right/B-left/B-right independently belong to their own shoulder/armhole: no shared/branching sleeve, swallowed upper arm or missing endpoint. Under CONTACT ELIGIBILITY, verify genuine mutual contact and, for separated sources, local surface contact leaving independent arm routes and upper-body contours readable. A failed route requires moving complete figures/contact location."

PAIR_CAST_PRIORITY_GATE = "PAIR CAST PRIORITY: Reconstruct exactly two distinct complete people, A=Image 2 and B=Image 3, the exclusive identity roster. Match each to its bound face in Image 1 before reading body/clothes/actions. Preserve both identities and gender presentations; never duplicate, merge, replace either or fall back to one figure. Exclude unaccepted partners, faceless/background bodies, bystanders, reflections, screen images and edge fragments, even when larger or suggesting a conventional couple. Infer no male/female partner stereotype."

PAIR_CONTACT_ELIGIBILITY_GATE = "CONTACT ELIGIBILITY — ARM ROUTES BEFORE CONTACT: Bind A/B to their own face anchors, clothes and four arm routes under ARM POSE AND HAND GEOMETRY. Count only contact demonstrably between these two accepted people; projected overlap, hidden hands and contact with excluded people are not evidence. ALREADY CONNECTED: Preserve supported contact participants, anatomical sides, hand arrangement and belongings; map the retained contact into the standing frame under ARM POSE AND HAND GEOMETRY without a new arm action. SEPARATED: Plan from the final camera view. Keep the upper bodies and ALL source-visible arm/hand routes separately readable, including the inner upper arms. First reserve unobstructed screen space for each source-visible wrist/hand and the sleeve leading to it. Place complete figures laterally with a small depth offset so a limited outer hip or lateral upper-thigh clothing surface touches, outside those arm routes. Lower-body surface contact satisfies connected composition; shoulders, armpits and sleeves need not touch. Never use overlapping white sleeves, one torso covering an inner arm, or a partner hiding a source-visible hand as the connection. Do not force a flush shoulder-to-hip seam or add a bridge/base/prop. Preserve visible leg stance; choose another local body-surface contact only if BOTH complete arm routes and all source-visible hands remain readable. No new handhold, linked arms, embrace or hand on the partner; no released phone hand or lowered outstretched arm. Original occluded segments may remain hidden only with a plausible route to their own shoulder; newly hiding a source-visible endpoint is a failed layout. Keep distinct garment edges, sleeve lengths and soft contact shadows. If the proposed contact conceals a required arm, discard that placement and recompose the complete figures before rendering. Anatomy and source-visible actions take priority over the preferred contact location."

PAIR_AMBIGUOUS_HAND_POLICY = "PAIR HAND EVIDENCE: Apply ARM POSE AND HAND GEOMETRY independently to all four owned hands; ambiguity never makes a hand available for a new connection."

PAIR_CONTACT_PREFERENCES = {'arm_contact': 'CONTACT PLAN: Apply CONTACT ELIGIBILITY exactly; keep original connections, otherwise use pose-preserving body-surface placement only.'}


def choose_my3d_pair_contact_preference() -> str:
    """Keep the existing route parameter while disabling randomized gestures."""
    return "arm_contact"


def build_my3d_connected_pair_prompt(
    wardrobe_profile: str | None = None,
    wardrobe_palette: str | None = None,
    *,
    upper_garment_profile: str | None = None,
    footwear_profile: str | None = None,
    source_context: str = "general",
    contact_preference: str | None = None,
) -> str:
    """Build the two-person prompt with one source ledger and compact audits."""
    return _assemble_my3d_human_prompt(
        cast_blocks=(PAIR_CAST_PRIORITY_GATE,),
        pose_blocks=(PAIR_CONTACT_ELIGIBILITY_GATE, POSE_AXIS_PROOF_GATE),
        character_blocks=(
            NON_PHOTOREALISTIC_FIGURINE_POLICY,
            FACE_DIRECTION_POLICY,
            "SOURCE-SUPPORTED KISS AND CONTACT: ONLY if both participants match the two accepted identity anchors, preserve their clearly visible face-to-face contact exactly; all contact with an unaccepted person is excluded, including a source-proven kiss with the two lips meeting at the same point, natural nose offset, eye closure, expression, head yaw and hand/arm contact. Never replace a kiss with nose-touching, a gap, or a generic embrace.",
            PAIR_AMBIGUOUS_HAND_POLICY,
            "UNSEEN PAIR WARDROBE: Design permitted missing areas independently under IDENTITY-BOUND BODY/UNSEEN OUTFIT. Never copy the partner's category, print/motif, fabric/cut, colorway, waistband, legwear, footwear or layer order. If the partner has a source-visible print, use a solid nonmatching fabric for an unseen garment.",
            HUMAN_FULL_FRAME_COMPOSITION_POLICY,
            "PAIR LAYOUT: Keep both anchored figures individually distinct, including heads, faces, hair, torsos and outfit sets. No collage, split panels, separate displays or floating figures. Make the eligible connection under CONTACT ELIGIBILITY plainly visible.",
            FIGURE_PROPORTION_FINAL_CHECK,
            HUMAN_NO_ADDED_DISPLAY_BASE_RULE,
        ),
    )

def append_my3d_human_prompt_audits(prompt: str, *, connected_pair: bool = False) -> str:
    """Keep exact final audit blocks once, after source and identity bindings."""
    final_blocks = [
        PAIR_HUMAN_OUTPUT_TEMPLATE_POLICY if connected_pair else HUMAN_OUTPUT_TEMPLATE_POLICY,
        FINAL_LOWER_BODY_FIDELITY_AUDIT,
        FINAL_UPPER_LAYER_AUDIT,
        FINAL_IDENTITY_COMPLETION_AUDIT,
    ]
    if connected_pair:
        final_blocks.append(PAIR_ANCHOR_ARM_OWNERSHIP_AUDIT)
    final_blocks.append(FIGURE_PROPORTION_FINAL_CHECK)
    final_blocks.append(HUMAN_HELD_OBJECT_HAND_POLICY)
    final_blocks.extend(_my3d_output_blocks())
    return _append_final_prompt_blocks(prompt, final_blocks)

def _my3d_output_blocks(*, pet: bool = False) -> tuple[str, ...]:
    """Single output-rule order for human and pet requests; lighting resolves last."""
    return (
        PET_OUTPUT_CLARITY_POLICY if pet else HUMAN_OUTPUT_CLARITY_POLICY,
        STUDIO_BACKGROUND_POLICY,
        PET_FULL_FRAME_COMPOSITION_POLICY if pet else HUMAN_FULL_FRAME_COMPOSITION_POLICY,
        PET_MATERIAL_COLOR_POLICY if pet else MATERIAL_COLOR_POLICY,
    )


def append_my3d_pet_prompt_audits(prompt: str) -> str:
    """Place the pet output rules once, after its identity anchor."""
    return _append_final_prompt_blocks(prompt, _my3d_output_blocks(pet=True))


def _append_final_prompt_blocks(prompt: str, final_blocks: list[str] | tuple[str, ...]) -> str:
    """Move exact owned rules to the tail; preserve reference/owner binding lines."""
    exact_blocks = set(final_blocks)
    kept_lines = []
    for line in prompt.splitlines():
        if line in exact_blocks:
            continue
        if line or (kept_lines and kept_lines[-1]):
            kept_lines.append(line)
    return "\n".join(kept_lines).rstrip() + "\n\n" + "\n".join(final_blocks)
