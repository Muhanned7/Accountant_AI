import axios from "axios";
import { useState } from "react";
import Panel from "./Panel.tsx";
function UploadBar() {
  const [setfile, setselectedfile] = useState<File | null>(null);
  const [Data, setData] = useState([]);
  const formData = new FormData();
  formData.append("file", setfile);
  const HandleUpload = async () => {
    console.log("hello Upload");
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/upload",
        formData,
        {
          headers: { Content: "multipart/form-data" },
        }
      );
      setData(response.data);
      console.log("File uploaded successfully:", response.data);
    } catch (error) {
      console.error("Upload error:", error);
    }
  };

  const HandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    console.log("hello change");
    if (event.target.files[0]) {
      setselectedfile(event.target.files[0]);
      setData([]);
    }
  };

  return (
    <>
      <input type="file" onChange={HandleChange} />
      <button onClick={HandleUpload}>Submit</button>
      <Panel data={Data}></Panel>
      <div>hello</div>
    </>
  );
}

export default UploadBar;
