import axios from "axios";
import { useState } from "react";

interface Props {
  Response: (data: string) => void;
}

function InputChat({ Response }: Props) {
  const [InputValue, setInputValue] = useState("");
  const [data, setData] = useState("");
  const formData = new FormData();
  formData.append("prompt", InputValue);
  const HandlePrompt = async (event: React.ChangeEvent<HTMLInputElement>) => {
    console.log(InputValue);
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/chatbot",
        { InputValue },
        { headers: { "Content-Type": "application/json" } }
      );
      Response(response.data);
      console.log(response.data);
    } catch (error) {
      console.error("Upload error:", error);
    }
  };
  const HandleChange = (event) => {
    setInputValue(event.target.value);
  };

  return (
    <>
      <input type="text" value={InputValue} onChange={HandleChange}></input>
      <button onClick={HandlePrompt}></button>
    </>
  );
}

export default InputChat;
