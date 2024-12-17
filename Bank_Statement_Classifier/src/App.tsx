import { useState } from "react";
import reactLogo from "./assets/react.svg";
import "./App.css";
import Message from "./components/Message.tsx";
import UploadBar from "./components/UploadBar.tsx";
import InputChat from "./components/InputChat.tsx";
import Response from "./components/Response.tsx";

function App() {
  const [data, setdata] = useState("");
  const ReceiveResponse = (childdata: string) => {
    setdata(childdata);
  };

  return (
    <>
      <div>
        <Response message={data}></Response>
      </div>
      <InputChat Response={ReceiveResponse} />
      <UploadBar />
    </>
  );
}

export default App;
