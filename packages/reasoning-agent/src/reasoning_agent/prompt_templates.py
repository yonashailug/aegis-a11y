"""Prompt templates for subject-specific LLM interactions.

Based on paper Section 3.2: Multi-modal Semantic Reasoning Layer
Templates designed for pedagogical alt-text generation following UDL guidelines.
"""

# Subject-specific prompt templates
STEM_TEMPLATES = {
    "physics": """
You are an expert in physics education and accessibility. Generate a pedagogical alt-text description for this physics element that focuses on learning objectives rather than visual appearance.

Context Information:
{context}

Guidelines:
- Focus on physical relationships, forces, energy, motion, and quantitative aspects
- Emphasize how the element supports physics learning objectives
- Use precise physics terminology while remaining accessible
- Highlight data trends, measurements, and scientific relationships
- Connect to physics principles and concepts

Element to describe: {element_text}
Subject area: {subject}

Provide a description that:
1. Explains the physics concepts being illustrated
2. Describes quantitative relationships and trends
3. Connects to learning objectives
4. Uses appropriate physics vocabulary

Description:""",
    "chemistry": """
You are an expert in chemistry education and accessibility. Generate a pedagogical alt-text description for this chemistry element that focuses on molecular understanding and chemical processes.

Context Information:
{context}

Guidelines:
- Focus on molecular structures, chemical reactions, and atomic interactions
- Emphasize bonding patterns, reaction mechanisms, and chemical properties
- Use chemistry terminology accurately while maintaining accessibility
- Highlight chemical relationships and transformations
- Connect to chemistry principles and concepts

Element to describe: {element_text}
Subject area: {subject}

Provide a description that:
1. Explains the chemical concepts being illustrated
2. Describes molecular structures and interactions
3. Connects to chemistry learning objectives
4. Uses appropriate chemical terminology

Description:""",
    "biology": """
You are an expert in biology education and accessibility. Generate a pedagogical alt-text description for this biology element that focuses on biological processes and living systems.

Context Information:
{context}

Guidelines:
- Focus on biological structures, functions, and processes
- Emphasize organism interactions, cellular mechanisms, and life processes
- Use biology terminology appropriately while remaining accessible
- Highlight biological relationships and systems
- Connect to biological principles and concepts

Element to describe: {element_text}
Subject area: {subject}

Provide a description that:
1. Explains the biological concepts being illustrated
2. Describes structures and their functions
3. Connects to biology learning objectives
4. Uses appropriate biological terminology

Description:""",
    "mathematics": """
You are an expert in mathematics education and accessibility. Generate a pedagogical alt-text description for this mathematics element that focuses on mathematical reasoning and problem-solving.

Context Information:
{context}

Guidelines:
- Focus on mathematical relationships, patterns, and problem-solving strategies
- Emphasize numerical patterns, geometric properties, and algebraic relationships
- Use mathematical terminology precisely while maintaining clarity
- Highlight mathematical concepts and reasoning
- Connect to mathematical learning objectives

Element to describe: {element_text}
Subject area: {subject}

Provide a description that:
1. Explains the mathematical concepts being illustrated
2. Describes patterns, relationships, and problem-solving approaches
3. Connects to mathematics learning objectives
4. Uses appropriate mathematical terminology

Description:""",
}

HUMANITIES_TEMPLATES = {
    "history": """
You are an expert in history education and accessibility. Generate a pedagogical alt-text description for this history element that focuses on historical significance and contextual understanding.

Context Information:
{context}

Guidelines:
- Focus on historical significance, cause-effect relationships, and chronological context
- Emphasize the importance of events, people, and developments
- Use historical terminology appropriately while remaining accessible
- Highlight historical connections and impacts
- Connect to historical thinking and learning objectives

Element to describe: {element_text}
Subject area: {subject}

Provide a description that:
1. Explains the historical concepts being illustrated
2. Describes significance and contextual relationships
3. Connects to history learning objectives
4. Uses appropriate historical terminology

Description:""",
    "literature": """
You are an expert in literature education and accessibility. Generate a pedagogical alt-text description for this literature element that focuses on literary analysis and interpretation.

Context Information:
{context}

Guidelines:
- Focus on literary devices, themes, and symbolic meaning
- Emphasize character development, narrative structure, and literary techniques
- Use literary terminology appropriately while maintaining accessibility
- Highlight literary relationships and interpretations
- Connect to literature learning objectives

Element to describe: {element_text}
Subject area: {subject}

Provide a description that:
1. Explains the literary concepts being illustrated
2. Describes themes, symbols, and literary devices
3. Connects to literature learning objectives
4. Uses appropriate literary terminology

Description:""",
    "social_studies": """
You are an expert in social studies education and accessibility. Generate a pedagogical alt-text description for this social studies element that focuses on civic understanding and social concepts.

Context Information:
{context}

Guidelines:
- Focus on civic concepts, social systems, and cultural understanding
- Emphasize government, economics, geography, and social relationships
- Use social studies terminology appropriately while remaining accessible
- Highlight social connections and civic concepts
- Connect to social studies learning objectives

Element to describe: {element_text}
Subject area: {subject}

Provide a description that:
1. Explains the social studies concepts being illustrated
2. Describes civic or cultural significance
3. Connects to social studies learning objectives
4. Uses appropriate social studies terminology

Description:""",
}

GENERAL_TEMPLATES = {"default": """
You are an expert in educational accessibility. Generate a pedagogical alt-text description for this educational element that focuses on learning objectives and student understanding.

Context Information:
{context}

Guidelines:
- Focus on educational purpose and learning objectives
- Emphasize how the element supports student understanding
- Use clear, accessible language while maintaining educational value
- Highlight important concepts and relationships
- Connect to the overall learning context

Element to describe: {element_text}
Subject area: {subject}

Provide a description that:
1. Explains the educational concepts being illustrated
2. Describes the learning significance
3. Uses clear, accessible language
4. Connects to educational objectives

Description:"""}

# Few-shot learning examples for high-quality pedagogical alt-text
FEW_SHOT_EXAMPLES = {
    "physics_force_diagram": {
        "input": "Diagram showing forces acting on a block on an inclined plane",
        "output": "Physics force diagram illustrating three forces acting on a block positioned on an inclined plane: weight (mg) pointing vertically downward, normal force (N) perpendicular to the inclined surface, and friction force (f) parallel to the surface opposing motion. This diagram demonstrates the decomposition of forces and is essential for understanding force equilibrium and Newton's laws on inclined planes.",
    },
    "chemistry_molecular_structure": {
        "input": "Lewis structure of water molecule",
        "output": "Chemistry molecular structure showing the Lewis dot structure of water (H2O). The oxygen atom is in the center with two hydrogen atoms bonded to it, displaying bent molecular geometry with approximately 104.5-degree bond angle. Two lone pairs of electrons are shown on oxygen. This structure illustrates covalent bonding, electron pair geometry, and molecular polarity concepts essential for understanding water's properties.",
    },
    "biology_cell_diagram": {
        "input": "Diagram of plant cell with labeled organelles",
        "output": "Biology cell diagram illustrating a typical plant cell with major organelles labeled: cell wall providing structure, nucleus containing genetic material, chloroplasts for photosynthesis, mitochondria for cellular respiration, vacuole for storage, and endoplasmic reticulum for protein synthesis. This diagram demonstrates cellular organization and the relationship between structure and function in plant cells.",
    },
    "mathematics_graph": {
        "input": "Graph showing quadratic function y = x²",
        "output": "Mathematics graph displaying the quadratic function y = x² with characteristic parabolic shape opening upward. The vertex is located at the origin (0,0), representing the minimum point. The graph demonstrates symmetry about the y-axis and shows how y-values increase as x-values move away from zero in either direction, illustrating key properties of quadratic functions and parabolic relationships.",
    },
    "history_timeline": {
        "input": "Timeline of American Civil War events 1861-1865",
        "output": "History timeline illustrating major events of the American Civil War from 1861 to 1865. Key events include Fort Sumter attack (April 1861) marking the war's beginning, Emancipation Proclamation (January 1863) changing war aims, Gettysburg Battle (July 1863) as the turning point, and Lee's surrender at Appomattox (April 1865) ending the conflict. This timeline demonstrates the chronological progression and causal relationships between pivotal Civil War events.",
    },
}


# Template selection helper function
def get_template_for_subject(subject_area: str) -> str:
    """Get appropriate template for subject area."""

    # Check STEM subjects
    if subject_area in STEM_TEMPLATES:
        return STEM_TEMPLATES[subject_area]

    # Check Humanities subjects
    if subject_area in HUMANITIES_TEMPLATES:
        return HUMANITIES_TEMPLATES[subject_area]

    # Default template
    return GENERAL_TEMPLATES["default"]


def get_few_shot_example(subject_area: str, content_type: str = "") -> str:
    """Get relevant few-shot example for subject and content type."""

    # Look for specific examples
    example_key = f"{subject_area}_{content_type}"
    if example_key in FEW_SHOT_EXAMPLES:
        example = FEW_SHOT_EXAMPLES[example_key]
        return f"Example:\nInput: {example['input']}\nOutput: {example['output']}\n\n"

    # Look for subject-specific examples
    for key, example in FEW_SHOT_EXAMPLES.items():
        if subject_area in key:
            return (
                f"Example:\nInput: {example['input']}\nOutput: {example['output']}\n\n"
            )

    return ""  # No example found
