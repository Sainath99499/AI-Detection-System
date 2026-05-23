import { useLocation } from "react-router-dom";

import {
  CircularProgressbar,
  buildStyles
} from "react-circular-progressbar";

import "react-circular-progressbar/dist/styles.css";

const Results = () => {

  const location = useLocation();

  const result = location.state;

  if (!result) {

    return (

      <div className="min-h-screen flex items-center justify-center bg-slate-900 text-white">
        No results available
      </div>
    );
  }

  return (

    <div className="min-h-screen bg-slate-900 text-white p-10">

      {/* ========================================= */}
      {/* PAGE TITLE */}
      {/* ========================================= */}

      <h1 className="text-5xl font-bold mb-12">
        Detection Results
      </h1>

      {/* ========================================= */}
      {/* PROBABILITY CARDS */}
      {/* ========================================= */}

      <div className="grid md:grid-cols-2 gap-10">

        {/* AI PROBABILITY */}

        <div className="bg-slate-800 p-8 rounded-3xl shadow-lg">

          <h2 className="text-2xl mb-6 text-center">
            AI Probability
          </h2>

          <div className="w-52 h-52 mx-auto">

            <CircularProgressbar
              value={result.ai_probability}
              text={`${result.ai_probability}%`}
              styles={buildStyles({
                textColor: "#ffffff",
                pathColor: "#ef4444",
                trailColor: "#1e293b"
              })}
            />

          </div>

        </div>

        {/* HUMAN PROBABILITY */}

        <div className="bg-slate-800 p-8 rounded-3xl shadow-lg">

          <h2 className="text-2xl mb-6 text-center">
            Human Probability
          </h2>

          <div className="w-52 h-52 mx-auto">

            <CircularProgressbar
              value={result.human_probability}
              text={`${result.human_probability}%`}
              styles={buildStyles({
                textColor: "#ffffff",
                pathColor: "#22c55e",
                trailColor: "#1e293b"
              })}
            />

          </div>

        </div>

      </div>

      {/* ========================================= */}
      {/* RESULT SECTION */}
      {/* ========================================= */}

      <div className="bg-slate-800 p-10 rounded-3xl mt-12 shadow-lg">

        <h2 className="text-4xl font-bold mb-6">

          Prediction:
          {" "}

          <span className="text-blue-400">
            {result.prediction}
          </span>

        </h2>

        <h3 className="text-2xl mb-4">

          Confidence Level:
          {" "}

          <span className="text-yellow-400">
            {result.confidence}
          </span>

        </h3>

        {/* CONTENT TYPE */}

        {
          result.content_type && (

            <h3 className="text-xl mb-6 text-gray-300">

              Content Type:
              {" "}

              <span className="text-pink-400 capitalize">
                {result.content_type}
              </span>

            </h3>
          )
        }

        {/* MESSAGE */}

        <div className="mt-8">

          {
            result.ai_probability >
            result.human_probability

              ? (

                <p className="text-red-400 text-xl">
                  This content appears to be AI-generated.
                </p>
              )

              : (

                <p className="text-green-400 text-xl">
                  This content appears to be human-created.
                </p>
              )
          }

        </div>

      </div>

      {/* ========================================= */}
      {/* EXPLANATION ENGINE */}
      {/* ========================================= */}

      {
        result.explanation && (

          <div className="bg-slate-800 p-10 rounded-3xl mt-12 shadow-lg">

            <h2 className="text-3xl font-bold mb-6 text-cyan-400">
              AI Explanation Engine
            </h2>

            <p className="text-lg leading-8 text-gray-300">
              {result.explanation}
            </p>

          </div>
        )
      }

    </div>
  );
};

export default Results;