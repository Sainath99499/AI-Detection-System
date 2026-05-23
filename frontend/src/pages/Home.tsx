import { useNavigate } from "react-router-dom";

const Home = () => {

  const navigate = useNavigate();

  return (

    <div className="min-h-screen flex flex-col justify-center items-center bg-slate-900 text-white">

      <h1 className="text-5xl font-bold mb-6 text-center">
        AI Content Authenticity Detection
      </h1>

      <p className="text-xl text-gray-300 mb-10">
        Detect AI-generated text using DeBERTa AI
      </p>

      <button
        onClick={() => navigate("/upload")}
        className="bg-blue-600 px-8 py-4 rounded-2xl text-xl hover:bg-blue-700"
      >
        Start Detection
      </button>

    </div>
  );
};

export default Home;