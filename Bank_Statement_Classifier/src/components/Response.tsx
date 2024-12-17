import Message from "./Message";

interface Props {
  message: string;
}

function Response({ message }: Props) {
  return (
    <div>
      <p>{message}</p>
    </div>
  );
}
export default Response;
