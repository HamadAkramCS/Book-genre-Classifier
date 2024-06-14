import tkinter as tk
from tkinter import ttk, scrolledtext
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv('books_dataset.csv')

def preprocess_and_train(df):
    X = df['summary']
    y = df['genre']

    vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = vectorizer.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    with open('book_genre_classifier.pkl', 'wb') as f:
        pickle.dump((model, vectorizer, encoder), f)

    print("Model and other objects have been saved successfully.")

def predict_genre(description):
    vectorized_desc = vectorizer.transform([description])
    genre_encoded = model.predict(vectorized_desc)
    genre = encoder.inverse_transform(genre_encoded)[0]
    result_label.config(text=f'Predicted Genre: {genre}')


def load_model():
    global model, vectorizer, encoder
    with open('book_genre_classifier.pkl', 'rb') as f:
        model, vectorizer, encoder = pickle.load(f)

# Create the GUI
def create_gui():
    global result_label, rating_label, listbox

    root = tk.Tk()
    root.title("Book Genre Classifier")
    root.geometry("1000x700")
    root.configure(bg='#e0f7fa')

    style = ttk.Style()
    style.configure('TLabel', font=('Helvetica', 14), background='#e0f7fa')
    style.configure('TButton', font=('Helvetica', 10, 'bold'), background='#00796b', foreground='white')
    style.configure('TEntry', font=('Helvetica', 12))
    style.configure('TListbox', font=('Helvetica', 12))

    is_light_theme = True

    def toggle_color():
        nonlocal is_light_theme
        if is_light_theme:
            root.configure(bg='#263238')
            style.configure('TLabel', background='#263238', foreground='#ffffff')
            style.configure('TButton', background='#455a64', foreground='#ffffff')
            text_entry.configure(background='#37474f', foreground='#ffffff')
        else:
            root.configure(bg='#e0f7fa')
            style.configure('TLabel', background='#e0f7fa', foreground='#000000')
            style.configure('TButton', background='#00796b', foreground='#ffffff')
            text_entry.configure(background='#ffffff', foreground='#000000')
        is_light_theme = not is_light_theme

    color_button = ttk.Button(root, text="Toggle Color", command=toggle_color)
    color_button.pack(pady=10)

    alphabet_frame = ttk.Frame(root)
    alphabet_frame.pack(pady=10)

    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        btn = tk.Button(alphabet_frame, text=letter, command=lambda l=letter: filter_books_by_letter(l), width=3, bg='#00796b', fg='white')
        btn.pack(side=tk.LEFT, padx=2, pady=5)

    search_var = tk.StringVar()
    search_var.trace("w", lambda *args: update_list(search_var.get()))

    search_label = ttk.Label(root, text="Search Book:")
    search_label.pack(pady=10)

    search_entry = ttk.Entry(root, textvariable=search_var, width=50)
    search_entry.pack(pady=10)

    sort_label = ttk.Label(root, text="Sort Books By:")
    sort_label.pack(pady=10)

    sort_var = tk.StringVar(value="title")
    sort_combobox = ttk.Combobox(root, textvariable=sort_var, values=["title", "rating"], state="readonly")
    sort_combobox.pack(pady=10)
    sort_combobox.bind("<<ComboboxSelected>>", lambda event: sort_books(sort_var.get()))

    listbox = tk.Listbox(root, width=60, height=10, font=('Helvetica', 12))
    listbox.pack(pady=10)
    listbox.bind("<<ListboxSelect>>", lambda event: on_select(event, listbox))

    text_entry = scrolledtext.ScrolledText(root, height=8, width=80, font=('Helvetica', 12))
    text_entry.pack(pady=10)

    result_label = ttk.Label(root, text="Predicted Genre: ", font=('Helvetica', 14, 'bold'))
    result_label.pack(pady=10)

    rating_label = ttk.Label(root, text="Rating: ", font=('Helvetica', 14))
    rating_label.pack(pady=10)

    for title, rating in zip(df['title'], df['rating']):
        listbox.insert(tk.END, f"{title} (Rating: {rating})")

    def update_list(search_term):
        filtered_titles = df[df['title'].str.contains(search_term, case=False, na=False)][['title', 'rating']]
        listbox.delete(0, tk.END)
        for title, rating in zip(filtered_titles['title'], filtered_titles['rating']):
            listbox.insert(tk.END, f"{title} (Rating: {rating})")

    def on_select(event, listbox):
        try:
            index = listbox.curselection()[0]
            selected_title = listbox.get(index).split(" (Rating: ")[0]
            book_data = df[df['title'] == selected_title].iloc[0]
            description = book_data['summary']
            rating = book_data['rating']
            text_entry.delete('1.0', tk.END)
            text_entry.insert(tk.END, description)
            rating_label.config(text=f'Rating: {rating}')
            predict_genre(description)
        except IndexError:
            pass

    def sort_books(criteria):
        sorted_df = df.sort_values(by=criteria, ascending=(criteria == "title"))
        listbox.delete(0, tk.END)
        for title, rating in zip(sorted_df['title'], sorted_df['rating']):
            listbox.insert(tk.END, f"{title} (Rating: {rating})")

    def filter_books_by_letter(letter):
        filtered_titles = df[df['title'].str.startswith(letter)][['title', 'rating']]
        listbox.delete(0, tk.END)
        for title, rating in zip(filtered_titles['title'], filtered_titles['rating']):
            listbox.insert(tk.END, f"{title} (Rating: {rating})")

    root.mainloop()


def main():
    preprocess_and_train(df)
    load_model()
    create_gui()


if __name__ == "__main__":
    main()
