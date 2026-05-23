import { useState } from "react";

import { useNavigate } from "react-router-dom";

import API from "../services/api";

const Upload = () => {

  const navigate = useNavigate();

  // =========================================
  // STATES
  // =========================================

  const [text, setText] = useState("");

  const [image, setImage] = useState<File | null>(null);

  const [preview, setPreview] = useState("");

  const [audio, setAudio] = useState<File | null>(null);

  const [audioPreview, setAudioPreview] = useState("");

  const [video, setVideo] = useState<File | null>(null);

  const [videoPreview, setVideoPreview] = useState("");

  const [loading, setLoading] = useState(false);

  // =========================================
  // TEXT DETECTION
  // =========================================

  const handleTextDetection = async () => {

    if (!text.trim()) {

      alert("Please enter text");

      return;
    }

    try {

      setLoading(true);

      const response = await API.post(
        "/detect/text",
        {
          text: text
        }
      );

      navigate(
        "/results",
        {
          state: response.data
        }
      );

    } catch (error) {

      console.error(error);

      const msg = (error as any)?.response?.data || (error as any)?.message || "Text detection failed";

      alert(msg);

    } finally {

      setLoading(false);
    }
  };

  // =========================================
  // IMAGE CHANGE
  // =========================================

  const handleImageChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {

    const file = e.target.files?.[0];

    if (file) {

      setImage(file);

      setPreview(
        URL.createObjectURL(file)
      );
    }
  };

  // =========================================
  // IMAGE DETECTION
  // =========================================

  const handleImageDetection = async () => {

    if (!image) {

      alert("Please upload image");

      return;
    }

    try {

      setLoading(true);

      const formData = new FormData();

      formData.append(
        "file",
        image
      );

      const response = await API.post(
        "/detect/image",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data"
          }
        }
      );

      navigate(
        "/results",
        {
          state: response.data
        }
      );

    } catch (error) {

      console.error(error);

      const msg = (error as any)?.response?.data || (error as any)?.message || "Image detection failed";

      alert(msg);

    } finally {

      setLoading(false);
    }
  };

  // =========================================
  // AUDIO CHANGE
  // =========================================

  const handleAudioChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {

    const file = e.target.files?.[0];

    if (file) {

      setAudio(file);

      setAudioPreview(
        URL.createObjectURL(file)
      );
    }
  };

  // =========================================
  // AUDIO DETECTION
  // =========================================

  const handleAudioDetection = async () => {

    if (!audio) {

      alert("Please upload audio");

      return;
    }

    try {

      setLoading(true);

      const formData = new FormData();

      formData.append(
        "file",
        audio
      );

      const response = await API.post(
        "/detect/audio",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data"
          }
        }
      );

      navigate(
        "/results",
        {
          state: response.data
        }
      );

    } catch (error) {

      console.error(error);

      const msg = (error as any)?.response?.data || (error as any)?.message || "Audio detection failed";

      alert(msg);

    } finally {

      setLoading(false);
    }
  };

  // =========================================
  // VIDEO CHANGE
  // =========================================

  const handleVideoChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {

    const file = e.target.files?.[0];

    if (file) {

      setVideo(file);

      setVideoPreview(
        URL.createObjectURL(file)
      );
    }
  };

  // =========================================
  // VIDEO DETECTION
  // =========================================

  const handleVideoDetection = async () => {

    if (!video) {

      alert("Please upload video");

      return;
    }

    try {

      setLoading(true);

      const formData = new FormData();

      formData.append(
        "file",
        video
      );

      const response = await API.post(
        "/detect/video",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data"
          }
        }
      );

      navigate(
        "/results",
        {
          state: response.data
        }
      );

    } catch (error) {

      console.error(error);

      const msg = (error as any)?.response?.data || (error as any)?.message || "Video detection failed";

      alert(msg);

    } finally {

      setLoading(false);
    }
  };

  return (

    <div className="min-h-screen bg-slate-900 text-white p-10">

      <h1 className="text-5xl font-bold mb-12">
        AI Content Detection
      </h1>

      {/* ========================================= */}
      {/* TEXT DETECTION */}
      {/* ========================================= */}

      <div className="bg-slate-800 p-8 rounded-3xl mb-12">

        <h2 className="text-3xl mb-6">
          Text Detection
        </h2>

        <textarea
          className="w-full h-56 p-4 rounded-xl bg-slate-700 border border-gray-600"
          placeholder="Paste text here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <button
          onClick={handleTextDetection}
          disabled={loading}
          className="mt-6 bg-blue-600 px-6 py-3 rounded-xl hover:bg-blue-700"
        >
          {
            loading
              ? "Analyzing..."
              : "Detect AI Text"
          }
        </button>

      </div>

      {/* ========================================= */}
      {/* IMAGE DETECTION */}
      {/* ========================================= */}

      <div className="bg-slate-800 p-8 rounded-3xl mb-12">

        <h2 className="text-3xl mb-6">
          Image Detection
        </h2>

        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="mb-6"
        />

        {
          preview && (

            <img
              src={preview}
              alt="Preview"
              className="w-72 rounded-xl border border-gray-600 mb-6"
            />
          )
        }

        <button
          onClick={handleImageDetection}
          disabled={loading}
          className="bg-green-600 px-6 py-3 rounded-xl hover:bg-green-700"
        >
          {
            loading
              ? "Analyzing..."
              : "Detect AI Image"
          }
        </button>

      </div>

      {/* ========================================= */}
      {/* AUDIO DETECTION */}
      {/* ========================================= */}

      <div className="bg-slate-800 p-8 rounded-3xl mb-12">

        <h2 className="text-3xl mb-6">
          Audio Detection
        </h2>

        <input
          type="file"
          accept=".wav,audio/*"
          onChange={handleAudioChange}
          className="mb-6"
        />

        {
          audioPreview && (

            <audio
              controls
              className="mb-6"
            >
              <source src={audioPreview} />
            </audio>
          )
        }

        <button
          onClick={handleAudioDetection}
          disabled={loading}
          className="bg-purple-600 px-6 py-3 rounded-xl hover:bg-purple-700"
        >
          {
            loading
              ? "Analyzing..."
              : "Detect AI Audio"
          }
        </button>

      </div>

      {/* ========================================= */}
      {/* VIDEO DETECTION */}
      {/* ========================================= */}

      <div className="bg-slate-800 p-8 rounded-3xl">

        <h2 className="text-3xl mb-6">
          Video Detection
        </h2>

        <input
          type="file"
          accept="video/*"
          onChange={handleVideoChange}
          className="mb-6"
        />

        {
          videoPreview && (

            <video
              controls
              className="w-96 rounded-xl border border-gray-600 mb-6"
            >
              <source src={videoPreview} />
            </video>
          )
        }

        <button
          onClick={handleVideoDetection}
          disabled={loading}
          className="bg-red-600 px-6 py-3 rounded-xl hover:bg-red-700"
        >
          {
            loading
              ? "Analyzing..."
              : "Detect AI Video"
          }
        </button>

      </div>

    </div>
  );
};

export default Upload;