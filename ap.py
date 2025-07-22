import streamlit as st

# Set page config
st.set_page_config(
    page_title="The Great Curd Confusion",
    page_icon="🥣",
    layout="centered"
)

# Title
st.markdown("<h1 style='text-align: center; color: #d6336c;'>The Great Curd Confusion</h1>", unsafe_allow_html=True)

# Add image from GitHub
image_url = "https://github.com/prakhar141/medical/raw/main/ChatGPT Image Jul 22, 2025, 06_55_09 PM.png"
st.markdown(
    f"""
    <div style='text-align: center;'>
        <img src="{image_url}" alt="A tale from the lockdown kitchen" style="width: 400px; border-radius: 12px;" />
        <p style='color: gray; font-size: 14px;'>A tale from the lockdown kitchen</p>
    </div>
    """,
    unsafe_allow_html=True
)


# Subtitle / Intro
st.markdown(
    """
    <p style='text-align: center; font-size:18px; color: #6c757d;'>
    A Lockdown Laughter Tale That Proved Even Mistakes Can Set the Perfect Curd!
    </p>
    """,
    unsafe_allow_html=True
)

# Story Content
story = """
During the unsettling days of the first COVID-19 wave, when the world outside was still and silent, and the air heavy with worry, I found myself craving a moment of lightness. So here's a little tale from my own lockdown diary—something to tickle your funny bone, and maybe remind you of the small, silly joys that carried us through those uncertain times.

(And no, don’t be fooled by the title—this is not a recipe for setting curd!) 😄

It all began one quiet evening when I was caught up helping my son with his studies. Seeing me busy, my husband—bless his rare kitchen spirit—volunteered to prepare evening tea for the family, including a tall glass of cold coffee for our son. My father, equally enthusiastic, joined the kitchen crew to assist in this surprise tea-time mission.

Meanwhile, before joining my son at his desk, I had placed a dish of milk with a spoonful of curd on the kitchen platform to ferment and set into curd.

Soon enough, the tea was brewing with a rich aroma, and the cold coffee was ready to be served. My son took his first sip and instantly made a face: “Something’s off!” he said. I, like any trusting mother, brushed him off with a “Just drink it!”

But then I sipped my tea… and stopped. There was something strange about the taste.

That’s when I noticed a few lumps floating in the cold coffee. My curiosity turned to concern. I checked the flavor—oh no! It was unmistakably weird.

And then, like a scene from a detective film, it all came together in my head: The tea and coffee had both been made using the same milk I had set aside with curd to ferment! 😱

(Yes, that was the moment my inner Sherlock Holmes—powered by classic mom-instincts—snapped into action.)

When I asked the “guest chefs” (my husband and father) which milk they had used, they sheepishly pointed to the same bowl I had set aside earlier. They had unknowingly turned our beverages into curd-infused experimental brews.

Somehow, between awkward laughter and amused disbelief, we gulped down our oddly tangy drinks, turning the kitchen disaster into a memory to treasure.

But just when we thought the story had ended...

My ever-helpful father, on a post-tea cleaning mission, decided to put things away. He spotted the same curd-setting bowl (now half-used), and thoughtfully placed it in the fridge—thinking it was just plain milk.

Later that evening, while prepping dinner, I found the dish in the fridge. Without thinking much, I took it out and kept it back on the kitchen platform, assuming it would finish setting overnight.

And the next morning?

To my absolute delight and surprise, I found that the curd had set perfectly—as if it had never been part of this chaotic comedy at all.

In a time when the world stood still and every day felt uncertain, this tiny household mishap gave us the warmest laugh.

I hope this little story brought one to your face too. 😊
"""

# Display story text
st.markdown(f"<div style='font-size: 17px; line-height: 1.8;'>{story}</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 15px; color: #888;'>Written with love❤️by Archana Mathur during lockdown ✍️</p>",
    unsafe_allow_html=True
)
