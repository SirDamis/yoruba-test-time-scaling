from __future__ import annotations

from dataclasses import dataclass

from .examples import InferenceExample


@dataclass(frozen=True)
class Exemplar:
    """Few-shot demonstration for a Yoruba benchmark item.

    ``reasoning_en`` / ``reasoning_yo`` support E1 language-matched CoT.
    ``translated_question`` supports the translate-pivot strategy.
    """

    question: str
    choices: list[str] | None
    reasoning_en: str
    reasoning_yo: str
    answer: str
    translated_question: str


TASK_EXEMPLARS: dict[str, list[Exemplar]] = {
    "math": [
        Exemplar(
            question=(
                "Títí ní ìwé mẹ́ta (3). Ó ra àpò ìwé méjì (2) sí i. "
                "Àpò kọ̀ọ̀kan ní ìwé mẹ́rin (4). Ìwé mélòó ni Títí ní báyìí?"
            ),
            choices=None,
            reasoning_en=(
                "First translate the quantities: Titi starts with 3 books. "
                "She buys 2 more packs, and each pack has 4 books, so the new books are 2 * 4 = 8. "
                "Total books = 3 + 8 = 11."
            ),
            reasoning_yo=(
                "Kọ́kọ́ túmọ̀ iye náà: Títí bẹ̀rẹ̀ pẹ̀lú ìwé 3. "
                "Ó ra àpò ìwé 2 sí i, àpò kọ̀ọ̀kan sì ní ìwé 4, nítorí náà ìwé tuntun = 2 × 4 = 8. "
                "Àpapọ̀ ìwé = 3 + 8 = 11."
            ),
            answer="11",
            translated_question=(
                "Titi has 3 books. She buys 2 more packs of books. "
                "Each pack has 4 books. How many books does Titi have now?"
            ),
        ),
        Exemplar(
            question=(
                "Kọ̀mpútà mẹ́sàn-án (9) wà ní yàrá kan. Wọ́n fi kọ̀mpútà márùn-ún (5) kun un "
                "lojoojúmọ́ láti Ọjọ́ Ajé sí Ọjọ́ Ẹtì. Kọ̀mpútà mélòó ni yàrá náà ní ní Ọjọ́ Ẹtì?"
            ),
            choices=None,
            reasoning_en=(
                "First translate the time span: Monday through Friday is 5 days. "
                "Each day adds 5 computers, so added computers = 5 * 5 = 25. "
                "Total computers = 9 + 25 = 34."
            ),
            reasoning_yo=(
                "Kọ́kọ́ túmọ̀ àsìkò náà: láti Ọjọ́ Ajé sí Ọjọ́ Ẹtì jẹ́ ọjọ́ 5. "
                "Wọ́n fi kọ̀mpútà 5 kun un lojoojúmọ́, nítorí náà èyí tí a fi kun = 5 × 5 = 25. "
                "Àpapọ̀ kọ̀mpútà = 9 + 25 = 34."
            ),
            answer="34",
            translated_question=(
                "There are 9 computers in a room. They add 5 computers every day "
                "from Monday through Friday. How many computers are in the room on Friday?"
            ),
        ),
        Exemplar(
            question=(
                "Bọ́lá ní ṣokolétì mejilelọgbọn (32), arábìnrin rẹ̀ sì ní mejilelogoji (42). "
                "Tí wọ́n bá jẹ márundinlogoji (45), ṣokolétì mélòó ni yóò kù?"
            ),
            choices=None,
            reasoning_en=(
                "First combine both amounts: 32 + 42 = 74 chocolates. "
                "They eat 45 chocolates. Remaining chocolates = 74 - 45 = 29."
            ),
            reasoning_yo=(
                "Kọ́kọ́ ṣàpapọ̀ iye méjèèjì: 32 + 42 = 74 ṣokolétì. "
                "Wọ́n jẹ ṣokolétì 45. Ṣokolétì tó kù = 74 - 45 = 29."
            ),
            answer="29",
            translated_question=(
                "Bola has 32 chocolates, and her sister has 42. "
                "If they eat 45, how many chocolates remain?"
            ),
        ),
        Exemplar(
            question=(
                "Adé ní náírà ẹgbẹ̀rún méjì ó lé irinwó (2,400). Ó ná náírà ẹgbẹ̀ta (600) "
                "lórí ìwé. Ó pín owó tó kù sí ọ̀sẹ̀ mẹ́ta (3). Náírà mélòó ni yóò lò ní ọ̀sẹ̀ kọ̀ọ̀kan?"
            ),
            choices=None,
            reasoning_en=(
                "First subtract the amount spent: 2400 - 600 = 1800 naira remains. "
                "Then divide the remaining money equally across 3 weeks: 1800 / 3 = 600. "
                "So he can use 600 naira each week."
            ),
            reasoning_yo=(
                "Kọ́kọ́ yọ owó tí ó ná kúrò: 2400 - 600 = 1800 náírà ló kù. "
                "Lẹ́yìn náà, pín owó tó kù sí ọ̀sẹ̀ 3 dọ́gba: 1800 / 3 = 600. "
                "Nítorí náà, ó lè lò náírà 600 ní ọ̀sẹ̀ kọ̀ọ̀kan."
            ),
            answer="600",
            translated_question=(
                "Ade has 2,400 naira. He spends 600 naira on books. "
                "He divides the remaining money across 3 weeks. "
                "How many naira can he use each week?"
            ),
        ),
    ],
    "math_en": [
        Exemplar(
            question="Tom has 5 bags of apples. Each bag contains 8 apples. He gives 12 apples to his friend. How many apples does Tom have left?",
            choices=None,
            reasoning_en=(
                "Start with 5 bags * 8 apples per bag = 40 total apples. "
                "He gives away 12 apples. Remaining apples = 40 - 12 = 28."
            ),
            reasoning_yo=(
                "Bẹ̀rẹ̀ pẹ̀lú àpò 5 × èso ápù 8 nínú ọ̀kọ̀ọ̀kan = 40 èso lápapọ̀. "
                "Ó fún ọ̀rẹ́ rẹ̀ ní èso 12. Èso tó kù = 40 - 12 = 28."
            ),
            answer="28",
            translated_question="Tom has 5 bags of apples. Each bag contains 8 apples. He gives 12 apples to his friend. How many apples does Tom have left?",
        ),
        Exemplar(
            question="Sarah reads 25 pages of a book on Monday and 30 pages on Tuesday. The book has 120 pages. How many pages does she still need to read?",
            choices=None,
            reasoning_en=(
                "Pages read so far = 25 + 30 = 55. "
                "Remaining pages = Total pages - Pages read = 120 - 55 = 65."
            ),
            reasoning_yo=(
                "Ojú-ìwé tí a ti kà = 25 + 30 = 55. "
                "Ojú-ìwé tó kù = Àpapọ̀ - Èyí tí a ti kà = 120 - 55 = 65."
            ),
            answer="65",
            translated_question="Sarah reads 25 pages of a book on Monday and 30 pages on Tuesday. The book has 120 pages. How many pages does she still need to read?",
        ),
        Exemplar(
            question="A baker makes 6 trays of cookies. Each tray holds 24 cookies. He sells 85 cookies. How many cookies are left?",
            choices=None,
            reasoning_en=(
                "Total cookies = 6 trays * 24 cookies per tray = 144. "
                "Cookies left = 144 - 85 = 59."
            ),
            reasoning_yo=(
                "Àpapọ̀ kúkì = àtẹ̀ 6 × kúkì 24 nínú ọ̀kọ̀ọ̀kan = 144. "
                "Kúkì tó kù = 144 - 85 = 59."
            ),
            answer="59",
            translated_question="A baker makes 6 trays of cookies. Each tray holds 24 cookies. He sells 85 cookies. How many cookies are left?",
        ),
        Exemplar(
            question="James bought a bike for $320. He had $280 saved and earned the rest by doing chores. If he did chores for 5 weeks earning the same amount each week, how much did he earn per week from chores?",
            choices=None,
            reasoning_en=(
                "Amount needed from chores = 320 - 280 = 40. "
                "Per week earnings = 40 / 5 = 8."
            ),
            reasoning_yo=(
                "Owó tó nílò láti ọwọ́ iṣẹ́ = 320 - 280 = 40. "
                "Owó ọ̀sọ̀ọ̀sẹ̀ = 40 / 5 = 8."
            ),
            answer="8",
            translated_question="James bought a bike for $320. He had $280 saved and earned the rest by doing chores. If he did chores for 5 weeks earning the same amount each week, how much did he earn per week from chores?",
        ),
    ],
    # "qa": [
    #     Exemplar(
    #         question="Kínni ìṣesí ọrọ̀ ajé pàtàkì tó tàn kálẹ̀ jùlọ ní àgbáyé?",
    #         choices=["Ìwakùsà", "Ìdọdẹ àti ìkó ǹkan jọ", "Ẹja pípa ", "Iṣẹ́ Àgbẹ̀"],
    #         reasoning_en=(
    #             "The question asks about the most widespread economic activity globally. "
    #             "Agriculture sustains the largest share of humanity, especially in developing countries. "
    #             "The answer is D (Iṣẹ́ Àgbẹ̀)."
    #         ),
    #         reasoning_yo=(
    #             "Ìbéèrè náà ń béèrè nípa ìṣesí ọrọ̀ ajé tó tàn kálẹ̀ jùlọ ní àgbáyé. "
    #             "Iṣẹ́ àgbẹ̀ ni ó ń bọ́ ọ̀pọ̀ jùlọ ènìyàn, pàápàá ní àwọn orílẹ̀-èdè tó ń dàgbà sókè. "
    #             "Ìdáhùn ni D (Iṣẹ́ Àgbẹ̀)."
    #         ),
    #         answer="D",
    #         translated_question="What is the most widespread economic activity in the world?",
    #     ),
    #     Exemplar(
    #         question="Ẹ̀sìn wo ní ìsàlẹ̀ ni ẹ̀sìn àgbáyé?",
    #         choices=["Taoism", "Islam", "Shintoism", "Confucianism"],
    #         reasoning_en=(
    #             "World religions are those with large, globally distributed followings. "
    #             "Islam is a major world religion alongside Christianity, Hinduism, and Buddhism. "
    #             "Taoism, Shintoism, and Confucianism are primarily ethnic religions concentrated in East Asia. "
    #             "The answer is B."
    #         ),
    #         reasoning_yo=(
    #             "Àwọn ẹ̀sìn àgbáyé jẹ́ àwọn ẹ̀sìn tí ó ní àwọn ọmọlẹ́yìn púpọ̀ káàkiri àgbáyé. "
    #             "Islam jẹ́ ẹ̀sìn àgbáyé pàtàkì pẹ̀lú Christianity, Hinduism, àti Buddhism. "
    #             "Taoism, Shintoism, àti Confucianism jẹ́ àwọn ẹ̀sìn ẹ̀yà-pàtàkì tí ó wà ní East Asia. "
    #             "Ìdáhùn ni B."
    #         ),
    #         answer="B",
    #         translated_question="Which of the following is a world religion?",
    #     ),
    #     Exemplar(
    #         question="Èwo nínú àwọn wọ̀nyìí ni àpẹẹrẹ orílẹ̀-èdè tí ò ní ìpínlẹ̀?",
    #         choices=["Jámánì ", "Ísráẹ́lì ", "Palẹsitínì", "Románíà "],
    #         reasoning_en=(
    #             "A stateless nation is a people group without a sovereign state. "
    #             "Germany, Israel, and Romania are all recognised sovereign states. "
    #             "Palestine lacks full statehood in international law, making it a stateless nation. "
    #             "The answer is C."
    #         ),
    #         reasoning_yo=(
    #             "Orílẹ̀-èdè tí ò ní ìpínlẹ̀ jẹ́ ẹgbẹ́ ènìyàn kan tí kò ní ìjọba àṣẹ tìrẹ. "
    #             "Jámánì, Ísráẹ́lì, àti Románíà jẹ́ àwọn orílẹ̀-èdè tí a mọ̀ sí ẹlẹ́tọ̀ọ́. "
    #             "Palẹsitínì kò ní ipò ìjọba àṣẹ kíkún nínú òfin ilẹ̀ òkèèrè, èyí sọ ọ́ di orílẹ̀-èdè tí ò ní ìpínlẹ̀. "
    #             "Ìdáhùn ni C."
    #         ),
    #         answer="C",
    #         translated_question="Which of the following is an example of a stateless nation?",
    #     ),
    #     Exemplar(
    #         question="Ọ̀pọ̀lọpọ̀ àwọn orílẹ̀-èdè Latin America ti gba òmìnira",
    #         choices=["láìpé lẹ́yìn Ogun Àgbáyé II", "Ní ọdún 1960", "láàrín àsìkò Ogun Àgbáyé I ", "ní ìbẹ̀rẹ̀ ọgọ́rùn ọdún ìkankàn-din-lógún"],
    #         reasoning_en=(
    #             "Most Latin American nations achieved independence in the early 19th century "
    #             "through wars against Spanish and Portuguese colonial rule, between roughly 1810 and 1825. "
    #             "The other time periods are too late. The answer is D."
    #         ),
    #         reasoning_yo=(
    #             "Ọ̀pọ̀lọpọ̀ àwọn orílẹ̀-èdè Latin America gba òmìnira ní ìbẹ̀rẹ̀ ọgọ́rùn ọdún ìkankàn-din-lógún "
    #             "nípasẹ̀ àwọn ogun tako ìjọba amúnisìn Spain àti Portugal, láàárín nǹkan bíi 1810 sí 1825. "
    #             "Àwọn àsìkò yòókù ti pẹ́ jù. Ìdáhùn ni D."
    #         ),
    #         answer="D",
    #         translated_question="Most Latin American countries gained their independence",
    #     ),
    # ],
    
    "qa" : [
        Exemplar(
            question="Èwo nínú àwọn wọ̀nyí ni orísun agbára tí a lè tún ṣe?",
            choices=["Edu", "Òróró", "Agbára oòrùn", "Gáàsì àdánidá"],
            reasoning_en=(
                "The question asks which energy source is renewable. "
                "Solar energy is naturally replenished by the sun, whereas coal, oil, and natural gas are fossil fuels. "
                "Therefore, the answer is C."
            ),
            reasoning_yo=(
                "Ìbéèrè náà ń béèrè nípa orísun agbára tí a lè tún ṣe. "
                "Agbára oòrùn máa ń tún ara rẹ̀ ṣe nípasẹ̀ oòrùn, ṣùgbọ́n edu, òróró, àti gáàsì àdánidá jẹ́ epo ilẹ̀ tí kò ṣeé tún ṣe. "
                "Nítorí náà, ìdáhùn ni C ."
            ),
            answer="C",
            translated_question="Which of the following is a renewable source of energy?",
        ),
        Exemplar(
            question="Ẹ̀sìn wo ló dá lórí ẹ̀kọ́ Siddhartha Gautama?",
            choices=["Islam", "Buddhism", "Judaism", "Sikhism"],
            reasoning_en=(
                "Siddhartha Gautama, also known as the Buddha, founded Buddhism. "
                "The other religions were founded by different historical figures or developed through different traditions. "
                "Therefore, the answer is B."
            ),
            reasoning_yo=(
                "Siddhartha Gautama, tí a tún mọ̀ sí Buddha, ni ó dá ẹ̀sìn Buddhism sílẹ̀. "
                "Àwọn ẹ̀sìn mìíràn ní ìpilẹ̀ tàbí olùdásílẹ̀ tó yàtọ̀. "
                "Nítorí náà, ìdáhùn ni B."
            ),
            answer="B",
            translated_question="Which religion is based on the teachings of Siddhartha Gautama?",
        ),
        Exemplar(
            question="Èwo nínú àwọn wọ̀nyí ni àpẹẹrẹ orílẹ̀-èdè tó wà lórí erékùṣù?",
            choices=["Nepal", "Mongolia", "Madagascar", "Chad"],
            reasoning_en=(
                "An island country is surrounded by water. "
                "Madagascar is a large island nation in the Indian Ocean, while Nepal, Mongolia, and Chad are landlocked countries. "
                "Therefore, the answer is C."
            ),
            reasoning_yo=(
                "Orílẹ̀-èdè erékùṣù ni orílẹ̀-èdè tí omi yí ká. "
                "Madagascar jẹ́ orílẹ̀-èdè erékùṣù ní Òkun India, nígbà tí Nepal, Mongolia, àti Chad jẹ́ orílẹ̀-èdè tí kò ní etíkun. "
                "Nítorí náà, ìdáhùn ni C."
            ),
            answer="C",
            translated_question="Which of the following is an island country?",
        ),
        Exemplar(
            question="Ọ̀pọ̀ àwọn orílẹ̀-èdè Áfíríkà gba òmìnira nígbà wo?",
            choices=[
                "Ní ọgọ́rùn-ún ọdún kẹrìnlá",
                "Ní ọdún 1960s",
                "Lẹ́yìn Ogun Agbaye Kìnní",
                "Ní ọgọ́rùn-ún ọdún kẹrìndínlógún",
            ],
            reasoning_en=(
                "Many African countries became independent during the wave of decolonization in the late 1950s and 1960s. "
                "The 1960s are often called the 'Decade of Africa' because many countries gained independence then. "
                "Therefore, the answer is B."
            ),
            reasoning_yo=(
                "Ọ̀pọ̀ àwọn orílẹ̀-èdè Áfíríkà gba òmìnira ní àsìkò ìparí amúnisìn ní ìparí ọdún 1950 àti ọdún 1960. "
                "Ọdún 1960 ni a mọ̀ sí àsìkò tí ọ̀pọ̀ orílẹ̀-èdè Áfíríkà gba òmìnira. "
                "Nítorí náà, ìdáhùn ni B."
            ),
            answer="B",
            translated_question="Most African countries gained independence during which period?",
        ),
    ],

    "qa_en": [
        Exemplar(
            question="Which of the following is a renewable source of energy?",
            choices=["Coal", "Oil", "Solar energy", "Natural gas"],
            reasoning_en=(
                "The question asks which energy source is renewable. "
                "Solar energy is naturally replenished by the sun, whereas coal, oil, and natural gas are fossil fuels. "
                "Therefore, the answer is C (Solar energy)."
            ),
            reasoning_yo="",
            answer="C",
            translated_question="Which of the following is a renewable source of energy?",
        ),
        Exemplar(
            question="Which religion is based on the teachings of Siddhartha Gautama?",
            choices=["Islam", "Buddhism", "Judaism", "Sikhism"],
            reasoning_en=(
                "Siddhartha Gautama, also known as the Buddha, founded Buddhism. "
                "The other religions were founded by different historical figures or developed through different traditions. "
                "Therefore, the answer is B."
            ),
            reasoning_yo="",
            answer="B",
            translated_question="Which religion is based on the teachings of Siddhartha Gautama?",
        ),
        Exemplar(
            question="Which of the following is an island country?",
            choices=["Nepal", "Mongolia", "Madagascar", "Chad"],
            reasoning_en=(
                "An island country is surrounded by water. "
                "Madagascar is a large island nation in the Indian Ocean, while Nepal, Mongolia, and Chad are landlocked countries. "
                "Therefore, the answer is C."
            ),
            reasoning_yo="",
            answer="C",
            translated_question="Which of the following is an island country?",
        ),
        Exemplar(
            question="Most African countries gained independence during which period?",
            choices=[
                "The 14th century",
                "The 1960s",
                "After World War I",
                "The 16th century",
            ],
            reasoning_en=(
                "Many African countries became independent during the wave of decolonization in the late 1950s and 1960s. "
                "The 1960s are often called the 'Decade of Africa' because many countries gained independence then. "
                "Therefore, the answer is B."
            ),
            reasoning_yo="",
            answer="B",
            translated_question="Most African countries gained independence during which period?",
        ),
    ],
    "reading_comprehension": [
        Exemplar(
            question=(
                "Àyọkà:\nỌ̀jẹ̀ẹ́ Adéjùmọ́ máa ń kọ́ àwọn akẹ́kọ̀ọ́ ní ilé ẹ̀kọ́ alákọ̀ọ́bẹ̀rẹ̀. "
                "Ó ní ọmọ ọdún mẹ́tàlélógún (23) nínú kíláàsì rẹ̀.\n\n"
                "Ìbéèrè:\nKí ni iṣẹ́ Ọ̀jẹ̀ẹ́ Adéjùmọ́?"
            ),
            choices=["Oníṣẹ̀-ọ̀gbìn", "Olùkọ́", "Dókítà", "Awẹ̀"],
            reasoning_en=(
                "The passage states 'Ọ̀jẹ̀ẹ́ Adéjùmọ́ máa ń kọ́ àwọn akẹ́kọ̀ọ́', meaning she teaches "
                "students. So she is a teacher, which is option B (Olùkọ́)."
            ),
            reasoning_yo=(
                "Àyọkà sọ pé 'Ọ̀jẹ̀ẹ́ Adéjùmọ́ máa ń kọ́ àwọn akẹ́kọ̀ọ́', èyí túmọ̀ sí pé ó ń kọ́ni. "
                "Nítorí náà ó jẹ́ olùkọ́, àṣàyàn B (Olùkọ́)."
            ),
            answer="B",
            translated_question=(
                "Passage:\nMrs. Adejumo teaches students at a primary school. "
                "She has 23 students in her class.\n\nQuestion:\nWhat is Mrs. Adejumo's job?"
            ),
        ),
        Exemplar(
            question=(
                "Àyọkà:\nỌjà ńlá kan wà ní àárín ìlú. Lọ́jọ́ Ọjọ́bọ̀, àwọn olùtajà máa ń kó ọjà wá. "
                "Wọ́n máa ń ta ẹ̀fọ́, ẹja, àti ẹran.\n\n"
                "Ìbéèrè:\nỌjọ́ wo ni àwọn olùtajà máa ń kó ọjà wá?"
            ),
            choices=["Ọjọ́ Ajé", "Ọjọ́ Ìṣẹ́gun", "Ọjọ́bọ̀", "Ọjọ́ Ẹtì"],
            reasoning_en=(
                "The passage says 'Lọ́jọ́ Ọjọ́bọ̀, àwọn olùtajà máa ń kó ọjà wá' "
                "(On Thursday, traders bring goods). The answer is C (Ọjọ́bọ̀)."
            ),
            reasoning_yo=(
                "Àyọkà sọ pé 'Lọ́jọ́ Ọjọ́bọ̀, àwọn olùtajà máa ń kó ọjà wá'. "
                "Nítorí náà ìdáhùn ni C (Ọjọ́bọ̀)."
            ),
            answer="C",
            translated_question=(
                "Passage:\nThere is a large market in the center of town. On Thursday, traders "
                "bring goods. They sell vegetables, fish, and meat.\n\n"
                "Question:\nOn which day do the traders bring goods?"
            ),
        ),
        Exemplar(
            question=(
                "Àyọkà:\nÓ jẹ́ ọmọ ọdún mẹ́wàá (10). Ó ní ìwé mẹ́rin: ìwé ìròyìn, ìwé àròsọ, "
                "ìwé ẹ̀kọ́ ìtàn, àti ìwé ẹ̀kọ́ ìmọ̀ ìjìnlẹ̀.\n\n"
                "Ìbéèrè:\nÌwé mélòó ni ọmọ náà ní?"
            ),
            choices=["Ìwé méjì", "Ìwé mẹ́rin", "Ìwé mẹ́fà", "Ìwé mẹ́jọ"],
            reasoning_en=(
                "The passage says 'Ó ní ìwé mẹ́rin' (He has 4 books), so the answer is B (Ìwé mẹ́rin)."
            ),
            reasoning_yo=(
                "Àyọkà sọ pé 'Ó ní ìwé mẹ́rin', nítorí náà ìdáhùn ni B (Ìwé mẹ́rin)."
            ),
            answer="B",
            translated_question=(
                "Passage:\nHe is 10 years old. He has four books: a newspaper, a storybook, "
                "a history textbook, and a science textbook.\n\n"
                "Question:\nHow many books does the child have?"
            ),
        ),
        Exemplar(
            question=(
                "Àyọkà:\nAdìẹ àgbààgbà máa ń pariwo ní àárọ̀ kùtùkùtù. "
                "Ariwo rẹ̀ máa ń jí gbogbo agbárílé.\n\n"
                "Ìbéèrè:\nKí ni Adìẹ àgbààgbà máa ń ṣe ní àárọ̀ kùtùkùtù?"
            ),
            choices=["Ó máa ń sùn", "Ó máa ń pariwo", "Ó máa ń jẹun", "Ó máa ń lọ sí ọjà"],
            reasoning_en=(
                "The passage says 'Adìẹ àgbààgbà máa ń pariwo ní àárọ̀ kùtùkùtù' "
                "(The old rooster crows early in the morning). The answer is B."
            ),
            reasoning_yo=(
                "Àyọkà sọ pé 'Adìẹ àgbààgbà máa ń pariwo ní àárọ̀ kùtùkùtù'. "
                "Nítorí náà ìdáhùn ni B."
            ),
            answer="B",
            translated_question=(
                "Passage:\nThe old rooster crows early in the morning. "
                "Its noise wakes the whole household.\n\n"
                "Question:\nWhat does the old rooster do early in the morning?"
            ),
        ),
    ],
}


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str

    def as_text(self) -> str:
        return f"System:\n{self.system}\n\nUser:\n{self.user}"


def render_prompt(example: InferenceExample, prompt_style: str) -> PromptBundle:
    """Render a prompt for an E1/E2 strategy.

    BoN / TTC sampling reuses the same E1 prompt styles with ``n > 1`` and
    majority-vote selection. ``best_of_n_cot`` is kept as a legacy alias for
    English CoT.
    """
    if prompt_style in {"english_cot", "best_of_n_cot"}:
        return render_english_cot_prompt(example)
    if prompt_style == "yoruba_cot":
        return render_yoruba_cot_prompt(example)
    if prompt_style == "translate_pivot":
        return render_translate_pivot_prompt(example)
    raise ValueError(f"Unsupported prompt_style: {prompt_style!r}")


def render_exemplar_block(task: str, *, reasoning_mode: str) -> str:
    """Render few-shot demonstrations."""

    exemplars = TASK_EXEMPLARS.get(task)
    if not exemplars:
        return ""

    blocks = ["Examples:"]

    for ex in exemplars:
        parts = [
            f"Question:\n{ex.question}",
        ]

        if ex.choices:
            parts.append(
                "Choices:\n" + "\n".join(format_choices(ex.choices))
            )

        if reasoning_mode == "translate":
            parts.extend(
                [
                    f"English translation:\n{ex.translated_question}",
                    f"Reasoning:\n{ex.reasoning_en}",
                    f"Final answer: {ex.answer}",
                ]
            )
        else:
            reasoning = (
                ex.reasoning_yo
                if reasoning_mode == "yo"
                else ex.reasoning_en
            )

            parts.extend(
                [
                    f"Reasoning:\n{reasoning}",
                    f"Final answer: {ex.answer}",
                ]
            )

        blocks.append("\n\n".join(parts))

    blocks.append("Now answer the following question using the same format.")

    return "\n\n---\n\n".join(blocks) + "\n\n"

def is_english_input(example: InferenceExample) -> bool:
    """True for translated benchmark variants whose questions are already English."""

    return example.source_dataset.endswith("_translate")


def common_system_prompt(example: InferenceExample) -> str:
    if is_english_input(example):
        return (
            "You are a careful reasoning assistant.\n\n"
            "Read the problem carefully, preserve the quantities and relationships, "
            "and solve only from the information given."
        )

    return yoruba_system_prompt()


def yoruba_system_prompt() -> str:
    return (
        "You are a careful reasoning assistant.\n\n"
        "Read the Yoruba problem carefully, preserve the quantities and relationships, "
        "and solve only from the information given."
    )


def question_label(example: InferenceExample) -> str:
    return "Question" if is_english_input(example) else "Question (Yoruba)"


def quantity_instruction(example: InferenceExample) -> str:
    if is_english_input(example):
        return "- First identify the quantities and whether they mean add, subtract, multiply, or divide."
    return "- First identify the Yoruba quantities and whether they mean add, subtract, multiply, or divide."


def render_english_cot_prompt(example: InferenceExample) -> PromptBundle:
    task_map = {"afrimgsm_translate": "math_en", "afrimmlu_translate": "qa_en"}
    task_for_exemplars = task_map.get(example.source_dataset, example.task)
    exemplar_block = render_exemplar_block(
        task_for_exemplars,
        reasoning_mode="en",
    )

    user = "\n\n".join(
        [
            f"{exemplar_block}{render_problem_block(example, question_label=question_label(example))}",
            render_answer_format(example),
            (
                "Instructions:\n"
                "- Use the examples only as formatting guides, not as source facts.\n"
                f"{quantity_instruction(example)}\n"
                "- Reason step by step in English.\n"
                "- Finish with exactly one line:\n"
                "Final answer: <answer>"
            ),
        ]
    )

    return PromptBundle(
        system=common_system_prompt(example),
        user=user,
    )

def render_yoruba_cot_prompt(example: InferenceExample) -> PromptBundle:
    exemplar_block = render_exemplar_block(
        example.task,
        reasoning_mode="yo",
    )

    user = "\n\n".join(
        [
            f"{exemplar_block}{render_problem_block(example, question_label=question_label(example))}",
            render_answer_format(example),
            (
                "Instructions:\n"
                "- Use the examples only as formatting guides, not as source facts.\n"
                f"{quantity_instruction(example)}\n"
                "- Reason step by step in Yoruba.\n"
                "- Finish with exactly one line:\n"
                "Final answer: <answer>"
            ),
        ]
    )

    return PromptBundle(
        system=yoruba_system_prompt(),
        user=user,
    )

def render_translate_pivot_prompt(example: InferenceExample) -> PromptBundle:
    exemplar_block = render_exemplar_block(
        example.task,
        reasoning_mode="translate",
    )

    user = "\n\n".join(
        [
            f"{exemplar_block}{render_problem_block(example, question_label='Question (Yoruba)')}",
            render_answer_format(example),
            (
                "Instructions:\n"
                "- Use the examples only as formatting guides, not as source facts.\n"
                "- Translate the Yoruba question into English.\n"
                "- Solve the translated question by reasoning step by step in English.\n"
                "- Finish with exactly one line:\n"
                "Final answer: <answer>"
            ),
        ]
    )

    return PromptBundle(
        system=yoruba_system_prompt(),
        user=user,
    )

def render_problem_block(
    example: InferenceExample,
    *,
    question_label: str = "Question",
) -> str:
    parts = [f"{question_label}:\n{example.question}"]

    if example.choices:
        parts.append(
            "Choices:\n" + "\n".join(format_choices(example.choices))
        )

    return "\n\n".join(parts)

def render_answer_format(example: InferenceExample) -> str:
    if example.answer_type == "choice":
        return (
            "Answer format:\n"
            "Final answer: <option letter>"
        )

    if example.answer_type == "number":
        return (
            "Answer format:\n"
            "Final answer: <number>"
        )

    return (
        "Answer format:\n"
        "Final answer: <concise Yoruba answer>"
    )

def format_choices(choices: list[str]) -> list[str]:
    formatted = []
    for index, choice in enumerate(choices):
        label = chr(ord("A") + index)
        stripped = choice.strip()
        if len(stripped) >= 2 and stripped[0].upper() == label and stripped[1] in {".", ")"}:
            formatted.append(stripped)
        else:
            formatted.append(f"{label}. {stripped}")
    return formatted
