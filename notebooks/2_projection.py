# %%
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.loader import CorpusLoader
from core.embedder import Embedder
from core.projection import ProjectionAnalyzer
# %%
# --- Initialize loaders ---
MultiLingMPNET = Embedder(model_name="intfloat/multilingual-e5-small")
sentimentLoader = CorpusLoader(text_col="text", label_col="label")
sentimentLoader.load_from_huggingface("chcaa/fiction4sentiment", split="train")
sentimentLoader.split_binary_train_continuous_test(positive_threshold=7, negative_threshold=3, train_size=0.6, random_state=42)
sentimentMPNET = MultiLingMPNET.embed(sentimentLoader.train_texts, cache_path="../data/embeddings/fiction4_train_MultiLingMPNET_neg3_pos7.csv")
sentimentMPNET["label"] = sentimentLoader.train_labels


# %%
klimaLoader = CorpusLoader(path="../data/TildesSuperData/expanded_speeches.csv", text_col="sentence", label_col="meeting_number")
klimaLoader.load_csv()
klimaLoader.df['sentence'] = klimaLoader.df['sentence'].apply(lambda s: f"query: {s}")
klimaMPNET = MultiLingMPNET.embed(klimaLoader.df, cache_path="../data/embeddings/Sentence_Klima.csv")
filtered=klimaMPNET.iloc[:, :-5].copy()
filtered["extra_column"] = klimaMPNET.iloc[:, -2].values
analyzer = ProjectionAnalyzer(matrix_concept=sentimentMPNET, matrix_project=filtered)
analyzer.project()

# Add sentiment:
result = klimaMPNET.iloc[:, -5:].copy()
result["sentiment"] = analyzer.projected_in_1D



result.head()

# %%
# Make refrence embeddings to get a sense of projection scale.

import pandas as pd
sentences = ["query: Det er godt.", "query: Det er okay.", "query: Det er dårligt."]
refrence_df = pd.DataFrame({'sentence': sentences, 'fill' : [1, 2, 3]})
refrence_MPNET = MultiLingMPNET.embed(sentences, cache_path="../data/embeddings/refrence_posneg")
refrence_MPNET["extra_column"] = ["pos","neu","neg"]
refrence_MPNET
analyzer = ProjectionAnalyzer(matrix_concept=sentimentMPNET, matrix_project=refrence_MPNET)
analyzer.project()
analyzer.projected_in_1D



# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Your thresholds
negative = analyzer.projected_in_1D[2]
neutral = analyzer.projected_in_1D[1]
positive = analyzer.projected_in_1D[0]


plt.figure(figsize=(10, 5))
sns.lineplot(data=result, x="year", y="sentiment", marker='o')
plt.title("Sentiment Over Time")
plt.xlabel("Year")
plt.ylabel("Sentiment Score")
plt.grid(True)
plt.tight_layout()
# Add horizontal lines
plt.axhline(y=positive, color='green', linestyle='--', label='Det er godt.')
plt.axhline(y=neutral, color='gray', linestyle='--', label='Det er okay')
plt.axhline(y=negative, color='red', linestyle='--', label='Det er dårligt.')

# Add legend if desired
plt.legend()

# Show plot
plt.show()

# %%
# --- Safety/Danger Axis --- 
dangerous_saetninger = [
    "query: Jeg gik alene gennem skoven midt om natten.",
    "query: Han håndterede kemikalier uden beskyttelsesudstyr.",
    "query: De krydsede vejen uden at se sig for.",
    "query: Vi hørte skud i det fjerne.",
    "query: Hun nærmede sig den vilde hund.",
    "query: Jeg klatrede uden sikkerhedssele.",
    "query: Børnene legede tæt på kanten af klippen.",
    "query: Der opstod ild i køkkenet.",
    "query: Vi kørte gennem stormen uden at stoppe.",
    "query: Han løb ud på isen, selvom den var tynd.",
    "query: Jeg var ved at blive fanget i en stærk strøm.",
    "query: De ignorerede advarslerne fra myndighederne.",
    "query: Der lå glasskår over hele gulvet.",
    "query: Hun gik ud til vejkanten i mørket.",
    "query: Jeg hørte nogen lirke med låsen på døren.",
    "query: De gik ind i det forladte hus om natten.",
    "query: Bilen mistede vejgrebet i svinget.",
    "query: Han rakte hånden ind i maskinen.",
    "query: Vi befandt os midt i et jordskælv.",
    "query: Hun satte sig bag rattet efter at have drukket.",
    "query: Jeg gik uden hjelm på byggepladsen.",
    "query: De tændte bål i tør skov.",
    "query: Der kom gnister ud af stikkontakten.",
    "query: Vi stod for tæt på kanten under bjergvandring.",
    "query: Han kastede fyrværkeri mod folk.",
    "query: Hun stak hånden ned i et mørkt hul.",
    "query: Jeg mistede fodfæstet på en glat stige.",
    "query: De svømmede langt ud uden redningsudstyr.",
    "query: Han gik ind i et område med radioaktivt affald.",
    "query: Vi blev jaget af en vred hund.",
    "query: Det er farligt."
]

safe_saetninger = [
    "query: Jeg sad og læste i min yndlingsstol.",
    "query: Hun gik en rolig tur i parken.",
    "query: Vi sad omkring lejrbålet og sang.",
    "query: Han lavede te i køkkenet.",
    "query: Børnene legede fredeligt i haven.",
    "query: Jeg slappede af i sofaen med en god bog.",
    "query: Vi gik tur langs stranden i solskin.",
    "query: Han fodrede ænderne ved søen.",
    "query: Jeg lavede mad med min familie.",
    "query: Hun sov trygt i sin seng.",
    "query: Vi lyttede til stille musik.",
    "query: Jeg sad på biblioteket og studerede.",
    "query: De spillede kort ved spisebordet.",
    "query: Vi kiggede på stjerner i baghaven.",
    "query: Han plantede blomster i haven.",
    "query: Vi holdt picnic i det grønne.",
    "query: Jeg gik i bad og nød det varme vand.",
    "query: Hun læste godnathistorie for sit barn.",
    "query: Vi sad sammen og snakkede i timevis.",
    "query: Han gik i kiosken efter is.",
    "query: Jeg tog et langt, varmt fodbad.",
    "query: Vi spillede brætspil indendørs.",
    "query: Hun tegnede med farvekridt i ro og mag.",
    "query: Jeg cyklede stille gennem nabolaget.",
    "query: De sad på en bænk og drak kaffe.",
    "query: Jeg nød solopgangen fra altanen.",
    "query: Vi pakkede gaver ind til jul.",
    "query: Hun lavede puslespil med sin bedstemor.",
    "query: Jeg satte mig ved vinduet og så regnen falde.",
    "query: De læste avis og drak te i stilhed.",
    "query: Det er trygt."
]
sentence = dangerous_saetninger+safe_saetninger
label = ["negative"] * 31 + ["positive"] * 31


# --- EMBED DANGERS/SAFET ---
dangerMPNET = MultiLingMPNET.embed(sentence, cache_path="../data/embeddings/DangerSafety.csv")
dangerMPNET["label"] = label
# --- Project onto vector (postivie = safe, negative = danger) ---
analyzer = ProjectionAnalyzer(matrix_concept=dangerMPNET, matrix_project=filtered)
analyzer.project()

# Add sentiment:
result = klimaMPNET.iloc[:, -5:].copy()
result["sentiment"] = analyzer.projected_in_1D
result

analyzer = ProjectionAnalyzer(matrix_concept=dangerMPNET, matrix_project=dangerMPNET)
analyzer.project()
refrence_danger = analyzer.projected_in_1D

# --- PLOT!!! ---

import seaborn as sns
import matplotlib.pyplot as plt

# Your thresholds
negative = refrence_danger[30]
positive = refrence_danger[61]


plt.figure(figsize=(10, 5))
sns.lineplot(data=result, x="year", y="sentiment", marker='o')
plt.title("Safety/Danger Over Time")
plt.xlabel("Year")
plt.ylabel("Safety to Danger axis:")
plt.grid(True)
plt.tight_layout()
# Add horizontal lines
plt.axhline(y=positive, color='green', linestyle='--', label= sentence[61])
plt.axhline(y=negative, color='red', linestyle='--', label= sentence[30])

# Add legend if desired
plt.legend()
# Show plot
plt.show()
















# %%
# --- Safety/Danger Axis --- 
pro_green_saetninger = [
    "query: Grøn energi er afgørende for at bekæmpe klimaforandringer.",
    "query: Jeg synes solenergi er en fantastisk løsning.",
    "query: Vindmøller er et vigtigt skridt mod en bæredygtig fremtid.",
    "query: Det er inspirerende at se byer satse på grøn strøm.",
    "query: Jeg støtter investeringer i vedvarende energikilder.",
    "query: Grøn energi skaber både jobs og en renere planet.",
    "query: Fremtiden tilhører sol- og vindenergi.",
    "query: Jeg vil gerne have mit hjem drevet af solceller.",
    "query: Elbiler kombineret med grøn strøm er vejen frem.",
    "query: Det giver mening at udfase kul og olie.",
    "query: Vedvarende energi gør os mindre afhængige af fossile brændstoffer.",
    "query: Jeg tror på, at grøn teknologi kan redde vores klima.",
    "query: Vindkraft er både effektivt og miljøvenligt.",
    "query: Jeg er glad for, at min kommune bruger grøn energi.",
    "query: Solenergi er blevet billigere og mere tilgængelig.",
    "query: Det er ansvarligt at satse på grøn omstilling.",
    "query: Grøn energi beskytter naturen og vores sundhed.",
    "query: Jeg vil hellere betale lidt mere for strøm, hvis den er grøn.",
    "query: Danmark er et foregangsland inden for vindenergi.",
    "query: Grøn omstilling skaber innovation og fremskridt.",
    "query: Jeg oplever, at grøn energi giver mening både økonomisk og etisk.",
    "query: Det er vigtigt at støtte klimavenlige løsninger.",
    "query: Jeg har installeret solceller og det virker fantastisk.",
    "query: Grøn energi er fremtidens energikilde.",
    "query: Vindmøller skæmmer ikke landskabet – de peger på fremtiden.",
    "query: Jeg har tillid til, at grøn teknologi vil forbedre vores samfund.",
    "query: Grøn strøm gør mig stolt som forbruger.",
    "query: Jeg vil gerne være med til at støtte grønne løsninger.",
    "query: Det er opløftende at se den teknologiske udvikling i grøn energi.",
    "query: Jeg ser grøn energi som en nødvendighed, ikke et valg."
]

anti_green_saetninger = [
    "query: Grøn energi virker dyr og ineffektiv i praksis.",
    "query: Jeg synes, vindmøller ødelægger landskabet.",
    "query: Grøn omstilling føles som en økonomisk byrde.",
    "query: Jeg stoler ikke på, at solceller kan dække vores energibehov.",
    "query: Udbygningen af grøn energi går for hurtigt og ukritisk.",
    "query: Jeg oplever, at elpriserne stiger pga. grøn energi.",
    "query: Der mangler gennemsigtighed i den grønne omstilling.",
    "query: Grøn teknologi bliver presset igennem uden debat.",
    "query: Jeg savner realistiske alternativer til fossile brændstoffer.",
    "query: Vindmøller støjer og generer beboere i nærheden.",
    "query: Grøn energi kræver enorme ressourcer at producere.",
    "query: Det er uretfærdigt, at almindelige borgere skal betale for den grønne omstilling.",
    "query: Jeg føler, at grøn energi er mere ideologi end løsning.",
    "query: Solceller virker ikke optimalt i det danske vejr.",
    "query: Der bliver investeret i grøn energi uden effektive resultater.",
    "query: Grøn omstilling går ud over industrien og arbejdspladser.",
    "query: Jeg tror ikke, vi kan undvære olie og gas helt.",
    "query: Grøn energi afhænger for meget af vejrforhold.",
    "query: Den grønne agenda bliver brugt til at indføre nye afgifter.",
    "query: Jeg savner realistiske beregninger bag grønne beslutninger.",
    "query: Nogle grønne løsninger virker som greenwashing.",
    "query: Omstillingen virker elitær og langt fra virkeligheden.",
    "query: Grøn energi kan ikke bære samfundet alene.",
    "query: Det er uklart, om klimaet reelt forbedres af den grønne omstilling.",
    "query: Jeg er skeptisk over for batterier og affald fra grøn teknologi.",
    "query: Grøn omstilling bør ikke ske på bekostning af forsyningssikkerheden.",
    "query: Elnettet er ikke klar til en total grøn overgang.",
    "query: Politikerne forstår ikke de praktiske konsekvenser af grøn energi.",
    "query: Jeg mener, grøn energi bør være frivillig – ikke tvunget.",
    "query: Grøn omstilling virker mest som symbolpolitik."
]
sentence = pro_green_saetninger + anti_green_saetninger
label = ["positive"] * 30 + ["negative"] * 30


# --- EMBED DANGERS/SAFET ---
fosilMPNET = MultiLingMPNET.embed(sentence, cache_path="../data/embeddings/GreenFosil.csv")
fosilMPNET["label"] = label
# --- Project onto vector (postivie = safe, negative = danger) ---
analyzer = ProjectionAnalyzer(matrix_concept=fosilMPNET, matrix_project=filtered)
analyzer.project()

# Add sentiment:
result = klimaMPNET.iloc[:, -5:].copy()
result["sentiment"] = analyzer.projected_in_1D

analyzer = ProjectionAnalyzer(matrix_concept=fosilMPNET, matrix_project=fosilMPNET)
analyzer.project()
refrence_danger = analyzer.projected_in_1D

# --- PLOT!!! ---

import seaborn as sns
import matplotlib.pyplot as plt

# Your thresholds
positive = refrence_danger[0]
negative = refrence_danger[41]



plt.figure(figsize=(10, 5))
sns.lineplot(data=result, x="year", y="sentiment", marker='o')
plt.title("Green/Fosil Over Time")
plt.xlabel("Year")
plt.ylabel("Safety to danger axis:")
plt.grid(True)
plt.tight_layout()
# Add horizontal lines
plt.axhline(y=positive, color='green', linestyle='--', label= sentence[0])
plt.axhline(y=negative, color='red', linestyle='--', label= sentence[41])

# Add legend if desired
plt.legend()
# Show plot
plt.show()

analyzer = ProjectionAnalyzer(matrix_concept=fosilMPNET, matrix_project=filtered)
analyzer.project()




# %%
agens_saetninger = [
    "query: Jeg rejste mig og gik direkte hen til døren.",
    "query: Hun tog ansvar for situationen uden tøven.",
    "query: Vi begyndte straks at rydde op.",
    "query: Han trådte frem og talte på gruppens vegne.",
    "query: Jeg greb muligheden, da den opstod.",
    "query: De handlede hurtigt og effektivt.",
    "query: Vi tog sagen i egen hånd.",
    "query: Jeg besluttede mig for at gøre en forskel.",
    "query: Hun pakkede tasken og tog af sted.",
    "query: Vi organiserede mødet med det samme.",
    "query: Han tog initiativ til at finde en løsning.",
    "query: Jeg meldte mig frivilligt.",
    "query: De tog affære, da de opdagede fejlen.",
    "query: Vi satte en plan i værk.",
    "query: Hun gik i gang med arbejdet med det samme.",
    "query: Jeg skrev en klage og sendte den ind.",
    "query: De skubbede projektet fremad.",
    "query: Jeg begyndte at forberede mig allerede i går.",
    "query: Han trænede hver dag for at nå sit mål.",
    "query: Jeg tog kontakt til lederen med det samme.",
    "query: Hun satte sig mål og arbejdede målrettet.",
    "query: Vi gik i dialog med naboerne.",
    "query: Jeg forhandlede om prisen.",
    "query: Han meldte sig ind i foreningen for at bidrage.",
    "query: Vi gjorde, hvad vi kunne, for at hjælpe.",
    "query: Jeg stillede mig til rådighed.",
    "query: Hun kastede sig ud i det uden frygt.",
    "query: Vi udarbejdede en strategi.",
    "query: Han reagerede med det samme.",
    "query: Jeg gik direkte i gang med opgaven."
]
passive_saetninger = [
    "query: Jeg blev siddende og gjorde ingenting.",
    "query: Hun ventede uden at tage initiativ.",
    "query: Vi så bare til, mens det hele skete.",
    "query: Han forblev tavs gennem hele mødet.",
    "query: Jeg lod det hele passere uden indgriben.",
    "query: De blev stående i baggrunden.",
    "query: Intet blev gjort, selvom problemet var tydeligt.",
    "query: Jeg følte mig lammet og ude af stand til at handle.",
    "query: Hun lod andre træffe beslutningen.",
    "query: Vi forblev passive hele dagen.",
    "query: Han sagde ikke et ord.",
    "query: Jeg afventede besked fra ledelsen.",
    "query: Intet skete i flere timer.",
    "query: Hun lå bare i sengen og stirrede op i loftet.",
    "query: Jeg gjorde ikke modstand.",
    "query: Vi lod situationen udvikle sig af sig selv.",
    "query: Han reagerede ikke på nyheden.",
    "query: Jeg følte mig fanget og uden handlemuligheder.",
    "query: Vi undlod at blande os.",
    "query: Hun lod sig trække med uden at sige fra.",
    "query: Jeg blev bare siddende og så på.",
    "query: De forholdt sig passive under hele mødet.",
    "query: Han lod det ske uden kommentar.",
    "query: Jeg havde ingen energi til at gøre noget.",
    "query: Intet blev ændret.",
    "query: Hun sad stille og så det hele ske.",
    "query: Jeg blev overvældet og kunne ikke reagere.",
    "query: Vi hørte det, men gjorde ingenting.",
    "query: Han blev tilbage uden at deltage.",
    "query: Jeg lod det hele glide forbi."
]
labels = ["positive"] * 30 + ["negative"] * 30
sentence = agens_saetninger + passive_saetninger

# --- EMBED DANGERS/SAFET ---
agencyMPNET = MultiLingMPNET.embed(sentence, cache_path="../data/embeddings/AgensPassiv.csv")
agencyMPNET["label"] = labels
# --- Project onto vector (postivie = agency, negative = passive) ---
analyzer = ProjectionAnalyzer(matrix_concept=agencyMPNET, matrix_project=filtered)
analyzer.project()

# Add sentiment:
result = klimaMPNET.iloc[:, -5:].copy()
result["sentiment"] = analyzer.projected_in_1D

analyzer = ProjectionAnalyzer(matrix_concept=agencyMPNET, matrix_project=agencyMPNET)
analyzer.project()
reference = analyzer.projected_in_1D

# --- PLOT!!! ---

import seaborn as sns
import matplotlib.pyplot as plt

# Your thresholds
positive = reference[6]
negative = reference[58]



plt.figure(figsize=(10, 5))
sns.lineplot(data=result, x="year", y="sentiment", marker='o')
plt.title("Agency Over Time")
plt.xlabel("Year")
plt.ylabel("Passive/Active axis:")
plt.grid(True)
plt.tight_layout()
# Add horizontal lines
plt.axhline(y=positive, color='green', linestyle='--', label= sentence[6])
plt.axhline(y=negative, color='red', linestyle='--', label= sentence[58])

# Add legend if desired
plt.legend()
# Show plot
plt.show()


# %%

# ---- OLIE TYPE SENTIMENT: ---
# Reuse filtered (or klimaMPNET) as base
combined_df = klimaMPNET.iloc[:, -5:].copy()

# Add pos/neg sentiment:
danger_analyzer = ProjectionAnalyzer(matrix_concept=dangerMPNET, matrix_project=filtered)
danger_analyzer.project()
combined_df["danger"] = danger_analyzer.projected_in_1D

# Green/Fossil axis (already done above, reuse that)
green_fossil_analyzer = ProjectionAnalyzer(matrix_concept=fosilMPNET, matrix_project=filtered)
green_fossil_analyzer.project()
combined_df["green_fossil"] = green_fossil_analyzer.projected_in_1D
import matplotlib.pyplot as plt
import seaborn as sns

# Group by year to compute mean sentiment and green/fossil scores
yearly_means = combined_df.groupby("year")[["danger", "green_fossil"]].mean().reset_index()

# Sort by year for line plotting
yearly_means = yearly_means.sort_values("year")

# Plot
plt.figure(figsize=(10, 8))

# Scatter: each year as a point
sns.scatterplot(data=yearly_means, x="danger", y="green_fossil", hue="year", palette="viridis", s=100)

# Line: temporal path
plt.plot(yearly_means["danger"], yearly_means["green_fossil"], color='gray', linestyle='--', alpha=0.5)

correlation = combined_df["danger"].corr(combined_df["green_fossil"])


# Add correlation as text on plot
plt.text(
    x=yearly_means["danger"].min(),  # lower-left corner
    y=yearly_means["green_fossil"].max(),  # top of the plot
    s=f"Pearson r = {correlation:.2f}",
    fontsize=12,
    ha="left",
    va="top",
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
)

# Labels and aesthetics
plt.title("Yearly Development: Danger vs. Green-Energy Stance Axis")
plt.xlabel("Danger (safe → dangerous)")
plt.ylabel("Green Energy Axis (anti-green → pro-green)")
plt.grid(True)
plt.tight_layout()
plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
















# %%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
from PIL import Image

fig, ax = plt.subplots(figsize=(10, 8))

# Compute correlation once
correlation = combined_df["sentiment"].corr(combined_df["green_fossil"])

# Function to update the plot for each frame (year)
def update(frame):
    ax.clear()
    
    # Data up to current frame
    data = yearly_means.iloc[:frame + 1]
    current_year = int(data["year"].iloc[-1])

    # Scatter
    sns.scatterplot(ax=ax, data=data, x="sentiment", y="green_fossil", hue="year", palette="viridis", s=100, legend=False)

    # Line
    ax.plot(data["sentiment"], data["green_fossil"], color='gray', linestyle='--', alpha=0.5)

    # Static correlation text (bottom-right)
    ax.text(
        x=yearly_means["sentiment"].max(),
        y=yearly_means["green_fossil"].min(),
        s=f"Pearson r = {correlation:.2f}",
        fontsize=11,
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
    )

    # Dynamic year label (top-center)
    ax.text(
        x=(yearly_means["sentiment"].max() + yearly_means["sentiment"].min()) / 2,
        y=yearly_means["green_fossil"].max(),
        s=f"Year: {current_year}",
        fontsize=16,
        weight="bold",
        ha="center",
        va="top"
    )

    # Axes labels and limits
    ax.set_title("Yearly Development: Sentiment vs. Green/Fossil Axis")
    ax.set_xlabel("Sentiment Axis (negative → positive)")
    ax.set_ylabel("Green vs. Fossil Axis (fossil → green)")
    ax.grid(True)
    ax.set_xlim(yearly_means["sentiment"].min() - 0.01, yearly_means["sentiment"].max() + 0.01)
    ax.set_ylim(yearly_means["green_fossil"].min() - 0.01, yearly_means["green_fossil"].max() + 0.01)

# Animate
ani = animation.FuncAnimation(fig, update, frames=len(yearly_means), interval=800)

# Save as GIF
ani.save("sentiment_vs_green_fossil.gif", writer="pillow", fps=1)

plt.close()

# %%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
from PIL import Image

fig, ax = plt.subplots(figsize=(10, 8))

# Compute correlation once
correlation = combined_df["danger"].corr(combined_df["green_fossil"])

# Function to update each frame (one year at a time)
def update(frame):
    ax.clear()
    
    data = yearly_means.iloc[:frame + 1]
    current_year = int(data["year"].iloc[-1])

    # Scatter points
    sns.scatterplot(ax=ax, data=data, x="danger", y="green_fossil", hue="year", palette="viridis", s=100, legend=False)

    # Temporal line
    ax.plot(data["danger"], data["green_fossil"], color='gray', linestyle='--', alpha=0.5)

    # Static correlation text (bottom-right)
    ax.text(
        x=yearly_means["danger"].max(),
        y=yearly_means["green_fossil"].min(),
        s=f"Pearson r = {correlation:.2f}",
        fontsize=11,
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
    )

    # Dynamic year text (top-center)
    ax.text(
        x=(yearly_means["danger"].min() + yearly_means["danger"].max()) / 2,
        y=yearly_means["green_fossil"].max(),
        s=f"Year: {current_year}",
        fontsize=16,
        weight="bold",
        ha="center",
        va="top"
    )

    # Axes labels and layout
    ax.set_title("Yearly Development: Danger vs. Green-Energy Stance Axis")
    ax.set_xlabel("Danger Axis (safe → dangerous)")
    ax.set_ylabel("Green Energy Axis (anti-green → pro-green)")
    ax.grid(True)
    ax.set_xlim(yearly_means["danger"].min() - 0.01, yearly_means["danger"].max() + 0.01)
    ax.set_ylim(yearly_means["green_fossil"].min() - 0.01, yearly_means["green_fossil"].max() + 0.01)

# Animate
ani = animation.FuncAnimation(fig, update, frames=len(yearly_means), interval=800)

# Save as GIF
ani.save("danger_vs_green_energy.gif", writer="pillow", fps=1)

plt.close()

