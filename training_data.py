"""
Training data for language detection.
Each language gets a chunk of representative text (~200-400 words).
Using varied, natural text so n-gram frequencies are realistic.
In a real system (like Google), this would be GIGABYTES of text per language.
Here we use enough to demonstrate the technique clearly.
"""

TRAINING_TEXT = {
    "english": """
        The quick brown fox jumps over the lazy dog. In the beginning, there was nothing
        but darkness and silence across the vast emptiness of space. Scientists have long
        wondered about the origins of the universe and how everything came to be. Many
        people enjoy reading books, watching movies, and spending time with their families
        on weekends. The weather today is quite pleasant, with a gentle breeze blowing
        through the trees. Technology continues to advance at a rapid pace, changing the
        way we live and work every single day. Education remains one of the most important
        tools for personal growth and societal development. Children learn best when they
        are encouraged to ask questions and explore the world around them.
    """,
    "spanish": """
        El rápido zorro marrón salta sobre el perro perezoso. En el principio, no había
        nada más que oscuridad y silencio en la vasta inmensidad del espacio. Los científicos
        se han preguntado durante mucho tiempo sobre los orígenes del universo y cómo todo
        llegó a existir. A muchas personas les gusta leer libros, ver películas y pasar
        tiempo con sus familias los fines de semana. El clima de hoy es bastante agradable,
        con una suave brisa que sopla entre los árboles. La tecnología continúa avanzando
        a un ritmo rápido, cambiando la forma en que vivimos y trabajamos todos los días.
    """,
    "french": """
        Le rapide renard brun saute par-dessus le chien paresseux. Au commencement, il n'y
        avait rien d'autre que l'obscurité et le silence dans la vaste immensité de l'espace.
        Les scientifiques se sont longtemps demandé quelles étaient les origines de l'univers
        et comment tout est arrivé à exister. Beaucoup de gens aiment lire des livres, regarder
        des films et passer du temps avec leur famille le week-end. Le temps aujourd'hui est
        assez agréable, avec une douce brise qui souffle à travers les arbres.
    """,
    "german": """
        Der schnelle braune Fuchs springt über den faulen Hund. Am Anfang gab es nichts
        außer Dunkelheit und Stille in der weiten Leere des Weltraums. Wissenschaftler haben
        sich lange gefragt, wie das Universum entstanden ist und wie alles zur Existenz kam.
        Viele Menschen lesen gerne Bücher, schauen Filme und verbringen Zeit mit ihrer Familie
        am Wochenende. Das Wetter heute ist recht angenehm, mit einer sanften Brise, die durch
        die Bäume weht. Die Technologie entwickelt sich rasant weiter und verändert unser Leben.
    """,
    "italian": """
        La volpe marrone veloce salta sopra il cane pigro. In principio, non c'era nient'altro
        che oscurità e silenzio nella vasta immensità dello spazio. Gli scienziati si sono
        chiesti per molto tempo quali fossero le origini dell'universo e come tutto sia venuto
        ad esistere. A molte persone piace leggere libri, guardare film e passare del tempo con
        le loro famiglie nei fine settimana. Il tempo oggi è piuttosto piacevole, con una brezza
        leggera che soffia tra gli alberi.
    """,
    "portuguese": """
        A rápida raposa marrom salta sobre o cão preguiçoso. No princípio, não havia nada
        além de escuridão e silêncio na vasta imensidão do espaço. Os cientistas há muito se
        perguntam sobre as origens do universo e como tudo passou a existir. Muitas pessoas
        gostam de ler livros, assistir filmes e passar tempo com suas famílias nos fins de
        semana. O tempo hoje está bastante agradável, com uma brisa suave soprando pelas árvores.
    """,
    "dutch": """
        De snelle bruine vos springt over de luie hond. In het begin was er niets anders dan
        duisternis en stilte in de uitgestrekte leegte van de ruimte. Wetenschappers vragen
        zich al lang af hoe het universum is ontstaan en hoe alles tot bestaan kwam. Veel
        mensen lezen graag boeken, kijken films en brengen tijd door met hun familie in het
        weekend. Het weer is vandaag erg prettig, met een zachte bries door de bomen.
    """,
    "swedish": """
        Den snabba bruna räven hoppar över den lata hunden. I begynnelsen fanns det ingenting
        annat än mörker och tystnad i universums vidsträckta tomhet. Vetenskapsmän har länge
        undrat över universums ursprung och hur allt kom till. Många människor tycker om att
        läsa böcker, titta på filmer och tillbringa tid med sina familjer på helgerna. Vädret
        idag är ganska trevligt, med en mild bris som blåser genom träden.
    """,
    "polish": """
        Szybki brązowy lis przeskakuje nad leniwym psem. Na początku nie było nic innego jak
        ciemność i cisza w ogromnej pustce kosmosu. Naukowcy od dawna zastanawiają się nad
        pochodzeniem wszechświata i jak wszystko zaczęło istnieć. Wiele osób lubi czytać
        książki, oglądać filmy i spędzać czas z rodziną w weekendy. Pogoda dzisiaj jest dość
        przyjemna, z delikatną bryzą wiejącą przez drzewa.
    """,
    "russian": """
        Быстрая коричневая лиса перепрыгивает через ленивую собаку. В начале не было ничего,
        кроме темноты и тишины в огромной пустоте космоса. Учёные долгое время размышляли о
        происхождении вселенной и о том, как всё начало существовать. Многие люди любят читать
        книги, смотреть фильмы и проводить время с семьёй по выходным. Погода сегодня довольно
        приятная, с лёгким ветерком, дующим сквозь деревья.
    """,
    "ukrainian": """
        Швидка коричнева лисиця перестрибує через ліниву собаку. На початку не було нічого,
        крім темряви і тиші у величезній порожнечі космосу. Вчені довгий час розмірковували
        про походження всесвіту і про те, як усе почало існувати. Багато людей люблять читати
        книги, дивитися фільми і проводити час з родиною у вихідні. Погода сьогодні досить
        приємна, з легким вітерцем, що дує крізь дерева.
    """,
    "greek": """
        Η γρήγορη καφέ αλεπού πηδάει πάνω από τον τεμπέλη σκύλο. Στην αρχή δεν υπήρχε τίποτα
        άλλο παρά σκοτάδι και σιωπή στην απέραντη αχανή έκταση του διαστήματος. Οι επιστήμονες
        αναρωτιούνται εδώ και καιρό για την προέλευση του σύμπαντος και πώς όλα ήρθαν στην
        ύπαρξη. Πολλοί άνθρωποι απολαμβάνουν να διαβάζουν βιβλία, να βλέπουν ταινίες και να
        περνούν χρόνο με τις οικογένειές τους τα Σαββατοκύριακα.
    """,
    "turkish": """
        Hızlı kahverengi tilki tembel köpeğin üzerinden atlar. Başlangıçta, uzayın geniş
        boşluğunda karanlık ve sessizlikten başka bir şey yoktu. Bilim adamları uzun zamandır
        evrenin kökenini ve her şeyin nasıl var olduğunu merak ediyorlar. Birçok insan kitap
        okumayı, film izlemeyi ve hafta sonları ailesiyle zaman geçirmeyi sever. Bugün hava
        oldukça hoş, ağaçların arasından esen hafif bir esinti var.
    """,
    "arabic": """
        الثعلب البني السريع يقفز فوق الكلب الكسول. في البداية، لم يكن هناك شيء سوى الظلام
        والصمت في الفراغ الشاسع للفضاء. تساءل العلماء منذ فترة طويلة عن أصل الكون وكيف نشأ
        كل شيء إلى الوجود. يحب الكثير من الناس قراءة الكتب ومشاهدة الأفلام وقضاء الوقت مع
        عائلاتهم في عطلات نهاية الأسبوع. الطقس اليوم لطيف جدًا، مع نسيم خفيف يهب بين الأشجار.
    """,
    "hebrew": """
        השועל החום המהיר מדלג מעל הכלב העצלן. בתחילה לא היה דבר מלבד חושך ושקט בריק העצום
        של החלל. מדענים תהו זה זמן רב על מקורות היקום וכיצד הכל בא לידי קיום. הרבה אנשים
        נהנים לקרוא ספרים, לצפות בסרטים ולהעביר זמן עם משפחותיהם בסופי שבוע. מזג האוויר היום
        נעים למדי, עם רוח קלה הנושבת בין העצים.
    """,
    "hindi": """
        तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है। शुरुआत में, अंतरिक्ष की विशाल शून्यता में
        अंधकार और सन्नाटे के अलावा कुछ नहीं था। वैज्ञानिक लंबे समय से ब्रह्मांड की उत्पत्ति और
        सब कुछ कैसे अस्तित्व में आया, इस बारे में सोचते रहे हैं। बहुत से लोग किताबें पढ़ना,
        फिल्में देखना और सप्ताहांत में अपने परिवार के साथ समय बिताना पसंद करते हैं।
    """,
    "japanese": """
        速い茶色のキツネが怠け者の犬を飛び越える。初めには、宇宙の広大な虚無の中に暗闇と静寂しか
        なかった。科学者たちは長い間、宇宙の起源とすべてがどのように存在するようになったのかを
        考えてきた。多くの人々は本を読んだり、映画を見たり、週末に家族と時間を過ごすことを楽しんで
        いる。今日の天気はかなり快適で、木々の間を穏やかな風が吹いている。
    """,
    "chinese": """
        敏捷的棕色狐狸跳过了懒惰的狗。起初，太空浩瀚的虚空中除了黑暗和寂静之外什么都没有。
        科学家们长期以来一直在思考宇宙的起源以及一切是如何开始存在的。许多人喜欢读书、看电影，
        并在周末与家人共度时光。今天的天气相当宜人，树林间吹着轻柔的微风。技术继续快速发展，
        改变着我们每天生活和工作的方式。
    """,
    "korean": """
        빠른 갈색 여우가 게으른 개를 뛰어넘는다. 처음에는 우주의 광대한 공허 속에 어둠과 침묵
        외에는 아무것도 없었다. 과학자들은 오랫동안 우주의 기원과 모든 것이 어떻게 존재하게
        되었는지에 대해 궁금해했다. 많은 사람들이 책을 읽고, 영화를 보고, 주말에 가족과 시간을
        보내는 것을 좋아한다. 오늘 날씨는 꽤 쾌적하며 나무 사이로 부드러운 바람이 불고 있다.
    """,
    "vietnamese": """
        Con cáo nâu nhanh nhảu nhảy qua con chó lười biếng. Ban đầu, không có gì khác ngoài
        bóng tối và sự im lặng trong khoảng không bao la của vũ trụ. Các nhà khoa học đã từ
        lâu thắc mắc về nguồn gốc của vũ trụ và mọi thứ đã hình thành như thế nào. Nhiều người
        thích đọc sách, xem phim và dành thời gian với gia đình vào cuối tuần.
    """,
    "indonesian": """
        Rubah coklat yang cepat melompati anjing yang malas. Pada awalnya, tidak ada apa-apa
        selain kegelapan dan keheningan di kehampaan luas alam semesta. Para ilmuwan telah
        lama bertanya-tanya tentang asal usul alam semesta dan bagaimana semuanya mulai ada.
        Banyak orang senang membaca buku, menonton film, dan menghabiskan waktu dengan keluarga
        mereka di akhir pekan. Cuaca hari ini cukup menyenangkan, dengan angin sepoi-sepoi.
    """,
    "thai": """
        สุนัขจิ้งจอกสีน้ำตาลที่ว่องไวกระโดดข้ามสุนัขที่ขี้เกียจ ในตอนแรกไม่มีอะไรเลยนอกจาก
        ความมืดและความเงียบในความว่างเปล่าอันกว้างใหญ่ของอวกาศ นักวิทยาศาสตร์สงสัยมานานแล้วว่า
        จักรวาลกำเนิดขึ้นมาได้อย่างไรและทุกสิ่งเริ่มมีอยู่ได้อย่างไร หลายคนชอบอ่านหนังสือ
        ดูหนัง และใช้เวลากับครอบครัวในวันหยุดสุดสัปดาห์
    """,
}