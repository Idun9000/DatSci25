# %%
import sys, os
import pandas as pd 
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


# Begin concept vector collection
concept_vectors = analyzer.concept_vector

result.head()
# %%
# Make refrence embeddings to get a sense of projection scale.
# import pandas as pd
# sentences = ["query: Det er godt.", "query: Det er okay.", "query: Det er dårligt."]
# refrence_df = pd.DataFrame({'sentence': sentences, 'fill' : [1, 2, 3]})
# refrence_MPNET = MultiLingMPNET.embed(sentences, cache_path="../data/embeddings/refrence_posneg")
# refrence_MPNET["extra_column"] = ["pos","neu","neg"]
# refrence_MPNET
# analyzer = ProjectionAnalyzer(matrix_concept=sentimentMPNET, matrix_project=refrence_MPNET)
# analyzer.project()
# analyzer.projected_in_1D



# %%
import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 5))
sns.lineplot(data=result, x="year", y="sentiment", marker='o')
plt.title("Sentiment Over Time")
plt.xlabel("Year")
plt.ylabel("Sentiment Score")
plt.grid(True)
plt.tight_layout()

# The centroid thresholds
negative = analyzer.negative_1D
positive = analyzer.positive_1D
# Add horizontal lines
plt.axhline(y=positive.item(), color='green', linestyle='--', label='Positive Centroid')
plt.axhline(y=negative.item(), color='red', linestyle='--', label='Negative Centroid')

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

# Add danger to list of concept vectors
concept_vectors = pd.concat([concept_vectors, analyzer.concept_vector], ignore_index=True)
# --- PLOT!!! ---

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
sns.lineplot(data=result, x="year", y="sentiment", marker='o')
plt.title("Safety/Danger Over Time")
plt.xlabel("Year")
plt.ylabel("Danger to Safety axis:")
plt.grid(True)
plt.tight_layout()

# The centroid thresholds
negative = analyzer.negative_1D
positive = analyzer.positive_1D
# Add horizontal lines
plt.axhline(y=positive.item(), color='green', linestyle='--', label='Safety Centroid')
plt.axhline(y=negative.item(), color='red', linestyle='--', label='Danger Centroid')

# Add legend if desired
plt.legend()
# Show plot
plt.show()

# %%
# --- ProGreen/AntiGreen Axis --- 
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
# --- Project onto vector ---
analyzer = ProjectionAnalyzer(matrix_concept=fosilMPNET, matrix_project=filtered)
analyzer.project()

# Add Danger/Safety to concept vector list:
concept_vectors = pd.concat([concept_vectors, analyzer.concept_vector], ignore_index=True)

# Add sentiment:
result = klimaMPNET.iloc[:, -5:].copy()
result["sentiment"] = analyzer.projected_in_1D

analyzer = ProjectionAnalyzer(matrix_concept=fosilMPNET, matrix_project=fosilMPNET)
analyzer.project()
refrence_danger = analyzer.projected_in_1D

# --- PLOT!!! ---

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
sns.lineplot(data=result, x="year", y="sentiment", marker='o')
plt.title("Pro-Green/Anti-Green Over Time")
plt.xlabel("Year")
plt.ylabel("Anti- to Pro-Green axis:")
plt.grid(True)
plt.tight_layout()

# The centroid thresholds
negative = analyzer.negative_1D
positive = analyzer.positive_1D
# Add horizontal lines
plt.axhline(y=positive.item(), color='green', linestyle='--', label='Pro-Green Centroid')
plt.axhline(y=negative.item(), color='red', linestyle='--', label='Anti-Green Centroid')

# Add legend if desired
plt.legend()
# Show plot
plt.show()

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

# --- EMBED Agens ---
agencyMPNET = MultiLingMPNET.embed(sentence, cache_path="../data/embeddings/AgensPassiv.csv")
agencyMPNET["label"] = labels
# --- Project onto vector (postivie = agency, negative = passive) ---
analyzer = ProjectionAnalyzer(matrix_concept=agencyMPNET, matrix_project=filtered)
analyzer.project()

# Add Agency to Vector List:
concept_vectors = pd.concat([concept_vectors, analyzer.concept_vector], ignore_index=True)

# Add sentiment:
result = klimaMPNET.iloc[:, -5:].copy()
result["sentiment"] = analyzer.projected_in_1D

analyzer = ProjectionAnalyzer(matrix_concept=agencyMPNET, matrix_project=agencyMPNET)
analyzer.project()
reference = analyzer.projected_in_1D

# --- PLOT!!! ---
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
sns.lineplot(data=result, x="year", y="sentiment", marker='o')
plt.title("Agency Over Time")
plt.xlabel("Year")
plt.ylabel("Passive to Active axis:")
plt.grid(True)
plt.tight_layout()

# The centroid thresholds
negative = analyzer.negative_1D
positive = analyzer.positive_1D
# Add horizontal lines
plt.axhline(y=positive.item(), color='green', linestyle='--', label='Active Centroid')
plt.axhline(y=negative.item(), color='red', linestyle='--', label='Passive Centroid')

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


# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.histplot(result["sentiment"], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Projections onto Sentiment Axis")
plt.xlabel("Sentiment Projection Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()



# %%
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(10, 5))
sns.histplot(result["sentiment"], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Projections onto Sentiment Axis")
plt.xlabel("Sentiment Projection Value")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# %%
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Manually specify the labels
labels = ["Sentiment", "Danger", "Green", "Agency"]

# Compute cosine similarity
similarity_matrix = cosine_similarity(concept_vectors)

# Create labeled DataFrame
similarity_df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(similarity_df, annot=True, cmap='coolwarm', vmin=0, vmax=1, square=True)
plt.title('Cosine Similarity Between Concept Vectors')
plt.tight_layout()
plt.show()


# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Step 1: Extract data
embeddings = klimaMPNET.iloc[:, :-5]
years = klimaMPNET.iloc[:, -5]  # column with year info

# Step 2: PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings)

# Store results
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
pca_df["year"] = years

# Step 3: Group by year
mean_pca_by_year = pca_df.groupby('year')[['PC1', 'PC2']].mean().reset_index()

# Step 4: Project concept vectors
concept_labels = ["Sentiment", "Danger", "Green", "Agency"]
concept_projected = pca.transform(concept_vectors)
concept_df = pd.DataFrame(concept_projected, columns=["PC1", "PC2"])
concept_df["label"] = concept_labels

# Step 5: Plot
plt.figure(figsize=(10, 8))

# Scatter of mean PCA points by year
sns.scatterplot(data=mean_pca_by_year, x='PC1', y='PC2', hue='year', palette='viridis', s=100)

# Track progression over years
plt.plot(mean_pca_by_year["PC1"], mean_pca_by_year["PC2"], color='grey', linestyle='-', linewidth=1, alpha=0.6)

# Annotate years
for _, row in mean_pca_by_year.iterrows():
    plt.text(row['PC1'], row['PC2'] + 0.002, str(int(row['year'])), fontsize=8)

# Add concept vectors as arrows
# Add concept vectors as infinite lines (through origin)
xlim = plt.xlim(-0.02, 0.025)
ylim = plt.ylim(-0.02, 0.02)
for _, row in concept_df.iterrows():
    pc1, pc2 = row["PC1"], row["PC2"]
    direction = np.array([pc1, pc2])
    norm = np.linalg.norm(direction)
    if norm == 0:
        continue  # skip zero vector
    direction = direction / norm  # unit vector

    # Extend in both directions
    line_length = max(xlim[1], ylim[1]) * 1.2
    x_vals = np.array([-line_length, line_length])
    y_vals = direction[1] / direction[0] * x_vals

    plt.plot(x_vals, y_vals, linestyle='--', label=row["label"])
    plt.text(direction[0]*line_length*1.05, direction[1]*line_length*1.05, row["label"],
             fontsize=10, weight='bold', color='black')

# Final plot touches
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Mean PCA Embedding per Year with Concept Vectors as Lines")
plt.axis("equal")
plt.legend(title="Concept", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% Cosine similarity of concept vectors in PCA space
# Recompute cosine similarity on the PCA-projected concept vectors
similarity_matrix_pca = cosine_similarity(concept_projected)

# Round similarity values for display
similarity_df_pca = pd.DataFrame(similarity_matrix_pca, index=concept_labels, columns=concept_labels)
similarity_df_pca = similarity_df_pca.round(3)

# Update labels to include PC1/PC2 coordinates
concept_df["label_with_coords"] = concept_df.apply(
    lambda row: f'{row["label"]}\n({row["PC1"]:.3f}, {row["PC2"]:.3f})', axis=1
)
label_with_coords = concept_df["label_with_coords"].values

# Re-index the DataFrame for heatmap display
similarity_df_pca.index = label_with_coords
similarity_df_pca.columns = label_with_coords

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_df_pca, annot=True, fmt=".3f", cmap='coolwarm', vmin=0, vmax=1, square=True)
plt.title('Cosine Similarity Between Concept Vectors (in PCA Space)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



# %%
