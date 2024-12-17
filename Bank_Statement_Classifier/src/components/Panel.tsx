interface Props {
  data: [];
}

function Panel({ data }: Props) {
  return (
    <>
      <table>
        <thead>
          <th>Date</th>
          <th>Amount</th>
          <th>Descritpion</th>
          <th>Type</th>
        </thead>
        <tbody>
          {data.map((row) => (
            <tr key={row.serial}>
              <td>{row.date}</td>
              <td>{row.amount}</td>
              <td>{row.description}</td>
              <td>
                <input type="text" defaultValue={row.type}></input>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div>hey There</div>
    </>
  );
}
export default Panel;
