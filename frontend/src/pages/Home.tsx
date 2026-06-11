
const Home = () => {
  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col items-center justify-center">

      <h1 className="text-6xl font-bold text-cyan-400 mb-6">
        VeriAI
      </h1>

      <p className="text-gray-300 text-xl mb-10">
        Professional AI Content Detection System
      </p>

      <a
        href="/detect"
        className="bg-cyan-500 hover:bg-cyan-600 px-8 py-4 rounded-xl text-lg font-semibold"
      >
        Start Detection
      </a>

    </div>
  );
};

export default Home;