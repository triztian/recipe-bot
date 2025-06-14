# Taxonomy of Failure Modes

## 1. Follow up Q
**Definition:** The bot asks a follow-up question after the recipe, violating the system prompt's instruction not to do so.
**Examples:**
- "The prompt says don't ask follow up questions, but it asks a follow up question at the end."  
(trace_20250613_095721_580213)
- "It asked a follow up question at the end after the recipe which I don't want."  
(trace_20250613_095755_205198)

---

## 2. Serving Size
**Definition:** The bot provides unclear or ambiguous serving size information, making it difficult for users to know if the recipe fits their needs.
**Examples:**
- "Not clear on serving size. It's for family presumably, so if it's a family of 4 should they double it? or is that enough for 4?"  
(trace_20250613_095755_205198)

---

## 3. Unclear Instructions
**Definition:** The bot gives instructions that are vague, incomplete, or require tools not commonly available, making the recipe hard to follow.
**Examples:**
- "Mentions tooling many won't have 'spiralizer' while it should have just stuck with the approachable one."
- "Doesn't say how long to cook eggs for and it's not specific"  
(trace_20250613_100217_438658)

---

## 4. Extra Info
**Definition:** The bot includes additional suggestions or components (e.g., side dishes) without clear instructions or time accounting, leading to confusion about what is required.
**Examples:**
- "The user asked for a keto lunch with 30 minutes max, and the beef skillet seems good. But at the bottom it then recommends serving with sliced avocado or a salad. It's unclear if those are intended to be part of the recipe or just suggestions if they want to do more. Will it feel incomplete without it? Is making them accounted for in the 30 minutes?"
- "It also doesn't give instructions for the salad, and anything it recommends should have instructions unless it's clearly stated as an optional thing that they could look up a recipe for."  
(trace_20250613_100328_862104)

---

## 5. (No axial code assigned)
**Definition:** Some traces have open coding notes but have not yet been assigned an axial (failure mode) code. These should be reviewed and categorized as the taxonomy evolves.
**Examples:**
- "Great job educating that beans are not keto and offering an alternative. However, it seems like maybe this is someone considering keto but really has beans on hand to use. It should follow up and clarify, so system prompt should be fixed for this case."  
(trace_20250613_095738_269008) 