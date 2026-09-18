"""Prompt builder for the My3DFigure direct GPT Image 2 flow."""


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

MY3D_PET_PROMPT_VERSION = "my3d-pet-v1"

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


MY3D_PROMPT_VERSION = "balanced-figure-v9.4"

FACE_PANEL_POLICY = """COLLAGE OVERRIDE: Use ONLY the clearest face-bearing panel for face, body pose, clothes and held items. Ignore all other panels, including back views of the same person. Never merge views; absent details are unseen."""

FIGURE_PROPORTION_POLICY = "FIGURE PROPORTIONS — OUTPUT GEOMETRY: First construct a balanced full-body collectible about 3.4 head units tall. The complete hair-top-to-chin head occupies 28-30% of hair-top-to-soles figure height, never over 32%; measure against the figure, not the canvas. Reserve the remaining 70-72% for a substantial neck, torso, pelvis, legs and shoes. Keep a readable waist and normal-length stylized limbs; never squeeze the body or inflate the chest to balance a large head. This SAME silhouette applies to full-body, torso, head-only, half-head and extreme close-up inputs, regardless of source crop or face-detection branch. Source head size in the frame is NOT anatomical scale. Rebuild at full-figure camera distance, not by extending a large source head downward. Preserve identity, age, hairstyle and proven pose while resizing head/hair together to this silhouette. No giant-head bobblehead, miniature torso, toddler body or compressed legs."

FIGURE_PROPORTION_FINAL_CHECK = "FINAL PROPORTION CHECK: Use moderately stylized animated-character proportions, NOT super-deformed proportions. The shoulders should remain broader than the skull; do not put a broad giant head on narrow miniature shoulders. Compare the entire head INCLUDING crown hair with the complete figure through soles: about 3.4 heads tall. If the head exceeds 32% of figure height, reduce head and hair together and rebalance the neck connection BEFORE returning; do not shrink the body, crop the shoes or add canvas padding as a substitute. This changes scale only, never source-supported pose, clothing or held items."

BODY_CONTOUR_POLICY = "BODY CONTOUR: Preserve genuine source-supported anatomy and age/gender, not stereotypes from hair/clothes. Unseen torso for a clearly male subject: modest natural pectorals, no invented rounded breast-like bulges or exaggerated muscles. With ambiguous evidence stays neutral and unaccentuated. Do not flatten or masculinize a source-supported female body. Natural fabric drape, not chest volume invented from folds. Invented stance: relaxed shoulders, ribs over pelvis; no thrust-forward chest or arched back."

CLOSE_PORTRAIT_SCALE_POLICY = "FACE-ONLY REFERENCE: Do not extend the screenshot downward with a tiny body. Discard close-lens head enlargement; retain identity, expression, hair, source-supported facing direction and visible details. Never turn an adult headshot into toddler anatomy."

HUMAN_ONLY_POLICY = "HUMAN-ONLY OUTPUT: No living animals, even owned, held, touched or ridden; exclude them from every object/contact preservation rule. Keep visible inanimate belongings. Convert sitting, crouching or riding to standing on the person's own feet; relax hands whose contact depended on a removed animal/support. Never replace an animal with another object."


def build_my3d_pet_prompt() -> str:
    """Create the isolated GPT-image prompt for one validated cat or dog."""
    return "\n".join([
        "Create exactly one cute, non-photorealistic full-body 3D cartoon pet collectible from Image 1. The subject is one real cat or dog, never a human, toy animal, second pet, or hybrid creature.",
        "PRESERVE PET IDENTITY: retain the animal's species, breed-like features when visible, fur color, markings, coat length and texture, ear shape and position, eye color, muzzle shape, nose color, body build, tail, expression, pose, and every clearly visible collar, tag, harness or clothing detail. Do not invent accessories or markings.",
        "STYLE: premium painted vinyl/resin cartoon figurine with sculpted fur masses, expressive painted eyes, rounded friendly forms, and clear animal anatomy. Make the whole pet equally stylized; never use photoreal fur, a pasted photographic face, or a live-animal render.",
        "ANATOMY: exactly one head, two ears, four legs, four paws, one tail when source-supported, and a natural animal silhouette. No human hands, human feet, clothing made for people, extra limbs, fused paws, duplicated tails, or distorted facial features.",
        "COMPOSITION: show the complete pet centered on a portrait 2:3 white studio canvas with a soft floor shadow. Preserve a source-supported sitting, standing, lying, or walking pose; if the lower body is hidden, complete it naturally without changing visible features. Remove scenery, text, watermarks, people, and other animals. Return only the single pet character image.",
    ])

HEAD_EVIDENCE_POLICY = "HEAD ORIENTATION — NON-NEGOTIABLE: Render head roll 0 degrees by default. Allow a sideways head tilt ONLY when ALL three facts are unequivocally visible: (1) head, neck and enough torso; (2) remove shared camera roll; (3) head axis still bends relative to its OWN neck/torso. If any fact is absent, ambiguous or aligned -> 0 degrees. A fully visible body, diagonally photographed person, sloping canvas, gaze, facial asymmetry, hair, eye line or shoulder line is never tilt evidence. Never infer a bend from a generated body."

NON_PHOTOREALISTIC_FIGURINE_POLICY = "STYLE — NON-PHOTOREALISTIC PHYSICAL FIGURINE: Make a cute premium painted vinyl/resin CARTOON toy. Stylize the FACE as strongly as the body: large expressive painted doll eyes, simplified sculpted nose/lips/cheeks, smooth matte toy skin and sculpted hair masses. No photographic skin pores, lifelike skin texture, pasted photo face or realistic human head on a toy body. Retain recognizable identity and age through simplified shapes, expression and hair, not photographic texture. Keep the moderate head scale specified above; cartoon style comes from sculpting and painted features, NOT an oversized skull. The result must read immediately as a manufactured stylized 3D collectible, never as a live person, photographic portrait, photorealistic skin or a realistic human render."


def _build_close_portrait_prompt(wardrobe_profile, wardrobe_palette, footwear_profile):
    """No randomized upper design is sent for a face-dominant reference.

    Garment category remains unknown unless the source proves it. A collar or
    neckline does not by itself prove a separate shirt rather than a dress.
    """
    lower_profile = {
        "dress": "skirt", "short_dress": "short_skirt", "long_dress": "long_skirt",
        "jumpsuit": "long_trousers", "romper": "shorts", "bodysuit": "shorts",
    }.get(wardrobe_profile, wardrobe_profile)
    lower = WARDROBE_COMPLETIONS.get(lower_profile or "", WARDROBE_COMPLETIONS["shorts"])
    palette = WARDROBE_PALETTES.get(wardrobe_palette or "", WARDROBE_PALETTES["pastel"])
    shoes = FOOTWEAR_STYLES.get(footwear_profile or "", FOOTWEAR_COMPLETIONS["shorts"])
    return "\n".join([
        HEAD_EVIDENCE_POLICY,
        FIGURE_PROPORTION_POLICY,
        "Create one recognizable, moderately proportioned full-body 3D cartoon figurine from Image 1. Later images, if any, are same-source detail crops only. Establish the complete body and legs first, then fit a modestly enlarged stylized head to that body.",
        NON_PHOTOREALISTIC_FIGURINE_POLICY,
        FACE_PANEL_POLICY,
        HUMAN_ONLY_POLICY,
        BODY_CONTOUR_POLICY,
        CLOSE_PORTRAIT_SCALE_POLICY,
        "HEAD POSE: Photo type/framing cannot prove or rule out a head bend. Remove global camera rotation from the whole person first. CONFIRMED TILT: only when head, neck and enough torso clearly show an unambiguous head-to-body bend, preserve its direction and degree without exaggeration. DEFAULT otherwise, including missing/unclear body context: head roll 0 degrees; never use an invented body as evidence. For DEFAULT only, align face/neck/sternum; frontal pupils level, nose/philtrum/chin vertical. Preserve facial asymmetry and gaze/up-down facing separately from roll. Hair, uneven eyes/shoulders or a cute pose alone cannot justify tilt. No invented lean, shoulder drop or hip shift; never tilt body/canvas to compensate. FACE YAW / VIEW DIRECTION: independently preserve the source's left/right facing direction and its degree. A side-facing or profile source MUST remain side-facing or profile in the generated character; never rotate it into a frontal face merely to show two eyes. If only one eye is visible in the source, render only the source-supported side of the face rather than inventing a symmetric front view.",
        "SOURCE CLOTHING FIRST: Treat visible color and visible construction as separate evidence. An unknown garment type does NOT make its color unknown. Preserve every visible fabric region's hue, lightness and saturation, neckline, shoulder coverage, sleeve, pattern, texture, trim and layers. Even a small edge of fabric is evidence. White stays white, black stays black; do not recolor either to match a palette or background. Studio lighting may shade the fabric but cannot change its base color. Do not add a collar, jacket, cutout or decorative texture over visible fabric.",
        "If only neckline/shoulders are visible, garment category below the crop remains unknown: a dress or other one-piece is allowed only with the SAME visible upper fabric, color and construction extended naturally. A proven separate top remains separate; a proven one-piece remains one-piece. If no clothing is visible at all, invent a coordinated outfit. Never erase or cover visible evidence for variety.",
        f"Only for missing separate LOWER clothing: {lower}. Only for missing lower clothing and shoes use {palette}. These suggestions have no authority over the upper garment or any visible region.",
        "If no lower clothing, legwear or footwear is visible, a clearly adult feminine presentation may optionally use coordinated hosiery: ankle, mid-calf, knee-high or full-length; black, white or matching. Never override visible evidence or an unclear age presentation.",
        f"Only for unseen footwear: {shoes}. Preserve any visible lower garment and footwear instead of these suggestions.",
        "Preserve face identity, expression, apparent age and gender presentation, skin tone, hairstyle and all clear worn/carried details, their counts, sides and contacts. Do not invent accessories or objects. Always a full-body standing figure. Remove shared camera roll before preserving proven body-relative bends. Unseen/unclear waist and pelvis -> upright torso, neutral hips/legs, no S-curve; do not turn looking down or reaching into sideways waist/neck bends. Head and torso need separate clear source evidence after de-roll; otherwise keep both upright. Unseen arms hang at the sides with relaxed empty hands; preserve any visible gesture or held item. Two arms/hands/legs/feet; each hand has one thumb and four fingers with natural occlusion, separate from objects. Rigid objects continue behind fingers; no finger passes through an object or branches at its edge. Do not mirror.",
        FIGURE_PROPORTION_FINAL_CHECK,
        "Complete figure through shoes centered on a portrait 2:3 white studio canvas with soft shadow, neutral color balance. Remove scenery, UI/text, other people and their body parts. FINAL STYLE CHECK: a non-photorealistic painted vinyl/resin cartoon collectible, never a live person or photo-real portrait. Human only, standing, head roll 0 degrees unless all three head-orientation facts above are proven; balanced head/body proportions. Return only the character image.",
    ])

# A single policy shared by every wardrobe/palette. This instructs the image
# model; it is not a local pose detector or a guarantee of generated geometry.
HEAD_POSE_POLICY = """HEAD POSE — highest priority over styling and all other pose instructions:
DEFAULT: head roll 0 degrees relative to its neck/torso; no invented lateral lean.
UNKNOWN: head-only or face-only photo, or neck/torso cropped, hidden, blurred or insufficient -> DEFAULT. Never invent a body and then use it as evidence that the source head was tilted. ALIGNED: the source head is aligned with its neck and torso -> DEFAULT. CONFIRMED TILT: only when the head, neck and enough torso are clearly visible AND show an unambiguous head-to-torso bend; preserve that relative direction and degree without exaggeration. A visible body alone does not qualify. If in doubt -> DEFAULT.
Remove global rotation from the whole person, never from the body alone. Camera roll, a sloping pupil line, facial/hair asymmetry or uneven shoulders alone are NOT proof of a neck bend. For DEFAULT, establish orientation before styling: align the facial midline with the neck and torso. For a frontal upright portrait, pupil centers share a horizontal line; nose bridge, philtrum and chin form a vertical axis. Preserve local facial asymmetry without amplifying tiny eye-height differences; judge by this axis, not the hair outline or parting. Preserve left/right gaze or up/down facing separately from lateral roll.
Only for invented body poses: level shoulders and a neutral stance; No cute lean, shoulder drop, hip shift or counter-tilt, even slightly. Preserve only proven body-relative torso action and steps.
FACE YAW / VIEW DIRECTION: preserve the exact left/right-facing direction and approximate yaw from the source independently of head roll. A side-facing or profile source MUST remain side-facing or profile in the generated character; never rotate it into a frontal face just to show two eyes. If the source shows only one eye, do not invent a symmetric front view."""

def build_my3d_direct_prompt(
    wardrobe_profile: str | None = None,
    wardrobe_palette: str | None = None,
    *,
    upper_garment_profile: str | None = None,
    footwear_profile: str | None = None,
    source_context: str = "general",
) -> str:
    """Compact universal source-fidelity prompt with conservative tilt permission."""
    if source_context == "close_portrait":
        return _build_close_portrait_prompt(wardrobe_profile, wardrobe_palette, footwear_profile)
    wardrobe = WARDROBE_COMPLETIONS.get(wardrobe_profile or "", WARDROBE_COMPLETIONS["shorts"])
    palette = WARDROBE_PALETTES.get(wardrobe_palette or "", WARDROBE_PALETTES["pastel"])
    colors = UNSEEN_COLOR_DIRECTIONS.get(wardrobe_palette or "", UNSEEN_COLOR_DIRECTIONS["pastel"])
    upper = UPPER_GARMENT_COMPLETIONS.get(upper_garment_profile or "", "a coordinated upper-garment design")
    footwear = FOOTWEAR_STYLES.get(footwear_profile or "", FOOTWEAR_COMPLETIONS.get(wardrobe_profile or "", FOOTWEAR_COMPLETIONS["shorts"]))
    return "\n".join([
        HEAD_EVIDENCE_POLICY,
        FIGURE_PROPORTION_POLICY,
        "Create exactly one clearly STYLIZED, moderately proportioned full-body 3D cartoon collectible figurine. Image 1 is the complete source, subject to the output rules below. Establish the complete body and legs first, then fit a modestly enlarged stylized head to that body. For a head-only or partial-head source, reconstruct a whole balanced character, not a giant portrait with an appended tiny body.",
        FACE_PANEL_POLICY,
        NON_PHOTOREALISTIC_FIGURINE_POLICY,
        HUMAN_ONLY_POLICY,
        HEAD_POSE_POLICY,
        BODY_CONTOUR_POLICY,
        "IDENTITY AND OUTPUT: Preserve face identity, expression, apparent gender/age presentation, skin tone and hairstyle. Do not turn a woman into a man or vice versa, or an adult into a child; Do not choose a stock child face. Use combined source evidence, not one hair/clothing cue. For distant/blurred sources, never default to a stock boy or masculine traits: retain coherent visible feminine presentation; if none is reliable, use a neutral adult presentation. No invented beard/makeup/sexualized features. Full figure through shoes on centered 2:3 white studio canvas with soft shadow.",
        "BODY POSE: Full-body output must stand naturally. Subject to HEAD POSE above, retain proven torso bends only after camera-roll removal; unseen/unclear waist/pelvis -> upright torso, neutral hips/legs, no S-curve. Head and torso need separate clear source evidence after de-roll; otherwise keep both upright. Preserve shoulder action, step, weight-bearing leg and hand/item relationships. Convert seated sources to standing; retain elbow bend, wrist/palm direction and hand height. Do not mirror; unseen arms relaxed/empty. Never replace a visible gesture or held item.",
        "OWNED DETAILS: Before styling, create a separate source-only object inventory for EACH hand: side, gesture, object count, orientation and contacts. Only unmistakable source-owned objects appear; empty hands stay empty. Never turn a faint line, highlight, strap edge, hair, shadow, seam or background mark into an object. Preserve simultaneous items, including partly hidden supported items, at source side/color/scale; also preserve clear headwear, eyewear, jewelry, watches, bags/straps/charms and tattoos/body markings. Never invent accessories or hide tattoos.",
        "HAND/OBJECT RENDER ORDER: Lock inventory and front/behind order. Rigid objects continue behind fingers; no finger passes through an object or branches at its edge. Uncertain blobs/streaks are not object types; unclear ownership means count zero.",
        "HAND ANATOMY: Two arms/hands/legs/feet. Each wrist has one palm, thumb plus four fingers. Show only source-exposed grip digits with natural finger occlusion; trace every fingertip to its own wrist/palm. No fused, extra or detached fingers; no duplicated, melted or object-like fingers. Preserve palm/thumb direction, separate object edges from skin, and never delete an item to simplify anatomy.",
        "FINAL HAND CHECK: five anatomical digits, source object count/contacts; no duplicates.",
        "VISIBLE CLOTHING/FOOTWEAR: Preserve EACH visible item and attribute: color, category, pattern, material, neckline, shoulder coverage, layers, sleeves, fasteners, seams, pockets, waistband, hem and fit. VISIBLE COLORS ARE LOCKED even small/rotated/shadowed; palettes, new layers and one-pieces never override/cover them. Preserve partial lower garments/shoes and legwear. Do not default to long trousers when hem unseen; only unknown attributes may vary.",
        "LEGWEAR EVIDENCE: Visible socks, hosiery, tights, bare legs and their connection to shoes are source evidence. Preserve their visible presence or absence, coverage, color and material; never add, remove or recolor them as a styling choice when any legwear evidence is visible.",
        "GARMENT CATEGORY: A neckline alone cannot prove top versus dress/jumpsuit/romper/bodysuit. Retain a proven dress/one-piece. A clear separate upper garment (shirt, blouse, tee or outer layer shown by collar/placket/sleeve/hem/layer boundary) stays separate in exact visible color/coverage; never turn it into or cover it with a dress/one-piece/new layer. Use a one-piece only when construction is continuous or category truly indeterminate without changing visible fabric. A small upper area keeps its exact color. No new cutouts/layers across visible areas.",
        f"UNSEEN COMPLETION ONLY: {wardrobe}. a dress selection becomes a skirt below it with a proven separate top; jumpsuit/romper selections become trousers/shorts; no authority over any visible item or attribute.",
        "UNKNOWN LOWER-BODY COMPLETION: Only when no usable lower-garment, legwear or footwear evidence is visible anywhere, a clearly adult feminine presentation may use coordinated hosiery as one optional completion. Vary short ankle, mid-calf, knee-high or full-length coverage and black, white, neutral or outfit-coordinated colors. This option never overrides visible evidence and must not be used for an unclear age presentation.",
        f"UNSEEN COLOR PLAN: {colors}. Style: {palette}. Visible colors override; unknown only.",
        f"UNSEEN FOOTWEAR ONLY: {footwear}. Coordinate outfit; retain all visible shoe attributes.",
        f"UNKNOWN UPPER DETAILS ONLY: {upper}. Unseen only; visible attributes fixed; one-pieces no extra layers.",
        FIGURE_PROPORTION_FINAL_CHECK,
        "CLEANUP: Remove scenery, UI/text, watermarks, other people/body parts and unowned objects. FINAL STYLE CHECK: a single non-photorealistic painted vinyl/resin cartoon collectible, never a live person or photo-real portrait; human only, standing, head roll 0 degrees unless all three head-orientation facts above are proven. Return character.",
    ])


def build_my3d_connected_pair_prompt(
    wardrobe_profile: str | None = None,
    wardrobe_palette: str | None = None,
    *,
    upper_garment_profile: str | None = None,
    footwear_profile: str | None = None,
    source_context: str = "general",
) -> str:
    """Build the isolated two-person prompt without changing the single prompt.

    The input is one checked image containing both people.  The output is one
    connected collectible composition, rather than two independent character
    renders or a collage.
    """
    return "\n".join([
        HEAD_EVIDENCE_POLICY,
        FIGURE_PROPORTION_POLICY,
        "Create exactly one polished cute 3D chibi collectible composition from Image 1, which contains exactly two distinct people. Both people must remain recognizable, complete standing figures with their own face, hair, clothing, arms, hands, legs and feet.",
        "TWO-PERSON COUNT GATE — HIGHEST PRIORITY: The checked source contains two usable human faces, so the output must contain both distinct people. Do not select a single primary person, omit the less-clear or partially occluded person, duplicate one person, merge two identities into one body, or fall back to a solo figure. Before styling, separately assign Person A and Person B using face, hair, clothing, body and object evidence; then render both assignments in the final image even when one source face is smaller, darker, blurred or partly hidden.",
        "CONNECTED TWO-PERSON MODE IS MANDATORY: The connected_pair request is never a one-person generation. Return one image containing both Person A and Person B; there is no valid single-person fallback, even if the image model finds one face easier to render.",
        "PERSON-SPECIFIC CLOTHING LEDGER — HIGHEST PRIORITY: Before posing or stylizing, build two separate clothing ledgers, one for each person. For each ledger, record only that person's visible garments and their owner, category, outer/inner layer order, neckline/collar, sleeve shape, closure, colour blocks, print, material, waistband/hem, legwear and footwear. Preserve every recorded attribute on its original owner. A multi-piece outfit must remain multi-piece; no visible layer may be simplified, merged, recoloured, removed, added, swapped or copied to the other person. This rule is universal: it applies to every colour, garment type, pattern, accessory and body region, never only to a particular outfit example.",
        "HIDDEN-CLOTHING INDEPENDENCE — HIGHEST PRIORITY: A garment visible on one person supplies zero evidence about the other person's hidden region. Complete only genuinely hidden regions for their own person, freely and plausibly, after the two clothing ledgers are locked. Give every completed unknown region an independent clothing fingerprint: garment category, pattern state, colour family, fabric, silhouette/hem/waist construction, legwear and footwear. Compare that fingerprint with every other person's visible clothing fingerprint. It must differ in at least three of those attributes; never reuse the other person's garment category, print, cut, hem, waistband, fabric, colourway, legwear, footwear or layer order as a template, continuation or matching set. If another person has any identifiable print or motif, an unknown region must use a solid, non-matching fabric rather than repeat that print. Only direct source evidence that both people match may override this rule. Reject and redesign any result that transfers or clones clothing across people.",
        "FINAL TWO-PERSON PRESENCE CHECK: Before returning, verify two visibly different heads/faces, two distinct hair masses, two torsos, two outfit sets and two connected full-body figures are present. A result showing one person, one head, one torso, or one person's clothing with a second face is invalid and must be corrected before output.",
        NON_PHOTOREALISTIC_FIGURINE_POLICY,
        "FACE SCALE AND EXPRESSION — SOURCE-LOCKED: Stylize the faces into a cute 3D cartoon, but keep each eye's size, spacing, eyelid shape and open/closed state proportional to that person's source face. Do not enlarge the eyes into oversized anime eyes, doll eyes or a generic chibi template; do not shrink the nose or mouth to compensate. Preserve each person's source expression and facial action before applying the cartoon material style.",
        HUMAN_ONLY_POLICY,
        HEAD_POSE_POLICY,
        "SOURCE-SUPPORTED KISS AND CONTACT — GENDER-NEUTRAL: If the source shows two clearly adult people kissing, reproduce that exact action regardless of whether they are two men, two women or a mixed-gender pair. Do not avoid, sanitize, weaken or reinterpret a same-sex or male-male kiss. The correct two mouths/lips must visibly meet at the same contact point; offset the noses naturally so the lips, not the noses, make contact. Preserve the source-supported lip closure, eye closure, head yaw, facial expression and hand/arm contact. Do not replace a kiss with nose-to-nose touching, smiling while looking at each other, an almost-kiss gap, puckered lips that do not touch, or a generic embrace. More generally, preserve every clearly visible face-to-face contact and its exact action before stylizing it.",
        "CONNECTED COMPOSITION — NON-NEGOTIABLE: The two people are one joined scene and one shared collectible composition. Their spatial distance and separate placement in the source are NOT locked. If the source shows them apart, actively move them together for the output while preserving each person's identity, face, hair, clothing, accessories, held items and person-specific colors. Use a natural shoulder-to-shoulder, linked-arm, hand-to-arm, hand-holding or gentle embrace pose so their bodies visibly touch or overlap at a believable contact zone. Do not preserve an empty gap merely because the source has one.",
        "SEPARATED-SOURCE RECOMPOSITION: A source with two distant or independently posed people must still become one physically connected pair in the final image. Re-pose only the spatial relationship and the minimum arms/shoulders needed to create natural contact; keep both people recognizable and keep clear source-owned objects readable and attached to the correct person. Never return two side-by-side figures with open space between them.",
        "POSE PRESERVATION WITH MINIMAL CONNECTION: Preserve each person's source head direction, gaze, expression, torso orientation, shoulder/hip alignment, leg stance, arm/elbow/wrist/palm pose, hand gesture, held object and clothing silhouette. For a separated source, first translate each complete figure inward as an intact unit until the nearest shoulders or upper arms touch; do not turn that translation into a new hug, pose swap, mirrored pose, changed phone grip or redesigned stance. Only make the smallest local shoulder/upper-arm/torso adjustment required to remove the gap, and never alter a source-supported gesture or object to manufacture contact.",
        "NO-NEW-HAND CONNECTION RULE — HIGHEST PRIORITY: Connect separated people by translating their intact bodies inward until shoulders or upper arms touch. This is the default and preferred connection. Do not create, borrow, duplicate or reroute a hand or forearm to make the connection. If the source does not show the two people holding hands or touching hands, the contact zone must contain zero newly introduced hands; do not turn an existing phone hand into a touching hand and do not make an arm terminate on the other person. A hand-to-hand or hand-on-body bridge is allowed only when that exact source-owned hand and contact are clearly present and can be traced continuously to its owner.",
        "VISIBLE CONTACT BRIDGE CHECK: The final image must show at least one unambiguous physical bridge between the two bodies, preferably the preserved nearest shoulders or upper arms touching after inward translation. Interlocked arms, one person's hand resting on the other's upper arm or waist, linked hands, or clear torso/hip contact are allowed only when they do not destroy a source-supported pose. Standing near each other, facing each other, aligned shoulders or overlapping shadows do NOT count. If both hands hold source-owned objects, use shoulder, upper-arm, torso or hip contact instead of inventing an extra hand. Never present them as two separate catalogue/model-sheet figures.",
        "Do not render two independent figures, two products, two panels, a diptych, a collage, a split image, or two floating characters. Use one shared base, one shared 2:3 portrait canvas, one lighting setup and one consistent chibi scale. The contact must read as intentional physical connection, not accidental overlap.",
        "COUNT AND IDENTITY: Render exactly two heads and two distinct faces, never a fused face or an extra face. Keep each person's apparent gender and age presentation, facial identity, expression, skin tone, hairstyle, gaze direction and visible asymmetry. Preserve which hair, clothing, accessory, hand and object belongs to which person; do not swap, merge or duplicate them.",
        "HAND/ARM COUNT AND OWNERSHIP — HARD LIMIT: Render exactly two arms and two hands per output person: four arms and four hands total, never a fifth hand. Every visible hand must trace continuously through its wrist and forearm to the shoulder/torso of one of these two people. Build a source-only inventory of both hands for each person, including held objects, side, wrist and contact; then render only those four owned hands. A hand or forearm must never emerge from the space between the people, a clothing edge, a hidden gap or the other person's torso. If the source contains a third person, remove that person's hands, arms and all other body parts completely; never borrow them to create contact.",
        "ANATOMY AT CONTACT: Arms and hands may naturally overlap or wrap around the other person only after their owner is established and the full limb remains plausibly attached. At the contact zone, prefer shoulder/upper-arm/torso occlusion with no added hand; use occlusion to hide an existing hand when needed, never invent a replacement hand. Preserve correct wrist and finger ownership, natural occlusion and believable attachment. Never fuse faces, heads, hands, feet or limbs, and never create detached or extra limbs to force the embrace.",
        "SOURCE POSE AND DETAILS: Use the combined image as the source of truth. Preserve each person's proven pose, clothing construction, visible colors, patterns, layers, footwear, headwear, eyewear, jewelry, bags, held items and body markings. If the source does not show a lower-body detail, complete it conservatively without covering visible evidence. Do not invent accessories or objects, and do not erase a clear source detail merely to simplify the joined pose.",
        "VISIBLE CLOTHING HARD LOCK — EACH PERSON SEPARATELY: Every clearly visible garment is source evidence, not a styling suggestion. For each person independently preserve the exact visible upper garment category, silhouette, neckline, shoulder coverage, sleeve/strap shape, hem, layering, fasteners, pattern, texture, color, lightness, saturation and material. A clearly visible black top stays that same black top; a clearly visible red top stays that same red top. Do not replace, recolor, lengthen, shorten, add a cardigan/jacket, add a new layer, change a top into a dress, or apply a generic fashion template when the upper garment is readable. Never copy Person A's visible garment onto Person B or use Person B's garment to fill Person A's uncertainty.",
        "CLOTHING COMPLETION BOUNDARY: Only genuinely hidden or indeterminate garment regions may be designed. Visible source garments, colors, patterns, legwear, hems and shoes always override creative completion. Keep every visible fabric region and its construction even when moving the people together; connection must not be solved by redesigning clothing.",
        "UNSEEN CLOTHING COMPLETION: Complete only genuinely hidden clothing for that same person. Never infer, extend, mirror, clone, or borrow the other person's visible clothes. Preserve every visible source garment on its owner; hidden clothing is free only after this per-person evidence check.",
        FIGURE_PROPORTION_FINAL_CHECK,
        "FINAL CLOTHING OWNERSHIP AUDIT: Recheck the two source ledgers after rendering. Every visible source garment must remain on its original owner with the same visible layers and attributes. For each completed hidden region, verify its clothing fingerprint differs from the other person's source clothing in at least three attributes and contains no copied print or motif. If either audit fails, replace only the offending hidden completion; do not alter the source-supported garment.",
        "Finish both figures through their shoes on one centered portrait 2:3 white studio canvas with soft floor shadow and neutral color balance. Remove scenery, UI/text, watermarks, other people and unowned objects. FINAL STYLE CHECK: exactly one non-photorealistic painted vinyl/resin cartoon collectible showing two connected people, never a live photo, never two separate figurines, never a separated pair. Return only the single joined character image.",
    ])
