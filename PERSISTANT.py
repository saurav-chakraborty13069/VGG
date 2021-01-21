"""
Author: Saurav Chakraborty
Version: 0.0.1
Date: 14-09-2020
"""


from flask import Flask, render_template, jsonify, request
import os
from flask_cors import CORS, cross_origin

app = Flask(__name__)


@app.route("/", methods=['POST'])
@cross_origin()
def TestString():
    input_str = request.json['inputstring']
    alphabets = list('abcdefghijklmnopqrstuvwxyz')
    alpha = []
    for char in alphabets:
        if char in list(input_str.lower()):
            alpha.append(char)

    if alpha == alphabets:
        result = True
    else:
        result =  False
    return jsonify(result)


if __name__ == "__main__":

    app.run(host='127.0.0.1', port=8000, debug=True)