from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
import pickle
import pandas as pd

from movie_model import build_model


# obj = build_model()

app = Flask(__name__)

app.config['SECRET_KEY'] = 'miniproject1'

class MovieInfo(FlaskForm):
    movie = StringField("Enter the movie name")
    search = SubmitField('Search')


@app.route('/', methods=['GET','POST'])
def index():
    data1 = pd.DataFrame()
    data2 = pd.DataFrame()
    data3 = pd.DataFrame()
    movie = False
    form = MovieInfo()
    if form.validate_on_submit():
        movie = form.movie.data
        data2 = build_model.get_recommendations(form.movie.data)
        # data2 = build_model.get_recommendations(form.movie.data).to_dict()
        data3 = build_model.display_movie(form.movie.data)
        data1 = build_model.top20
        form.movie.data = ''
        # dict1.a=
    # return render_template('index.html', form=form, movie=movie, data1=[data1.to_html(header="true", classes="data")], data2=[data2.to_html(header="True", classes="data")])
    return render_template('index.html', form=form, movie=movie, col1=data1.columns.values, col2=data2.columns.values,
                                row_data1=list(data1.values.tolist()), row_data2=list(data2.values.tolist()),
                                link_column="title", zip=zip, mcol = data3.columns.values, mrow=list(data3.values.tolist()))

# @app.route('/display', methods=['POST'])
# def display():
#
#
#     return render_template('display.html', )


if __name__ == '__main__':
    dat = pickle.load(open('model.pkl', 'rb'))
    app.run()
