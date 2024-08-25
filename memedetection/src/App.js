import React, { useEffect, useState } from 'react';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form';
import TextField from '@mui/material/TextField';
import { Button } from '@mui/material';
import ImageIcon from '@mui/icons-material/Image';
import axios from 'axios'

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [text,setText] = useState(null);
  const [response,setResponse] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && (file.type === 'image/png' || file.type === 'image/jpeg' || file.type === 'image/jpg')) {
      setSelectedFile(file);
    } else {
      setSelectedFile(null);
      alert('Please select a PNG, JPG, or JPEG file.');
    }
  };

  const handleSubmit = async () => {
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('text', text);

    try {
      axios.post('http://3.130.218.125:9000/detect_emotion', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      .then(response => {
        console.log(response.data);
        setResponse(response.data.predicted_labels)
      })
      .catch(error => {
        console.error('There was an error!', error);
      });
      
    }catch (e){
      console.log(e)
    }
  }

  useEffect(() => {
    console.log(selectedFile)
  },[selectedFile])
    

  return (
    <Container>
      <Row>
        <h1 style={{ fontWeight: '800', margin: '5vh 0' }}><center>Meme Emotion Detector</center></h1>
      </Row>
      <Row>
        <h3><center>Upload Meme</center></h3>
      </Row>
      <Row>
        <Container style={{ height: 'auto', backgroundColor: '#ffffff', borderRadius: '50px', display: 'flex', justifyContent: 'center', alignItems: 'center', flexDirection: 'column' }}>
          <Row xs={12}>
            <input type="file" accept=".png,.jpg,.jpeg" id="fileInput" style={{ display: 'none' }} onChange={handleFileChange} />
            <label htmlFor="fileInput">
              <Button variant="outlined" component="span" style={{ borderRadius: '50%', padding: '20px',marginTop:'3vh'}}>
                <ImageIcon style={{ fontSize: '10vh', color: 'black', cursor: 'pointer'}} />
              </Button>
            </label>
          </Row>
          <Row xs={12}>
            <Col><h3 style={{ color: 'black' }}>Upload the Image here</h3></Col>
          </Row>
          {
        selectedFile && 
      <Row>
        <img src={URL.createObjectURL(selectedFile)} style={{width:'20vw'}}/>
      </Row>
      }
          {/* <Row xs={12} style={{ width: '100%', padding: '0 15px', boxSizing: 'border-box', margin: '2vh' }}>
            <TextField
              fullWidth
              id="outlined-basic"
              label="Optional (Only enter if application is not able to detect the text)"
              value={text}
              onChange={(e) => {
                setText(e.target.value)
              }}
              variant="outlined"
              sx={{ borderRadius: '100px' }}
            />
          </Row> */}
          <Row xs={12} style={{ width: '100%', padding: '0 15px', boxSizing: 'border-box', margin: '2vh' }}>
            <Button onClick={() => {
              handleSubmit()
            }} variant='contained' style={{ borderRadius: '25px', height: '5vh' }}>Predict</Button>
          </Row>
        </Container>
      </Row>

      {response && <Container>
        <Row style={{textAlign:'center',margin:'5vh'}}>
          <h3 style={{marginBottom:'2rem'}}>Detected Persuasion Techniques </h3>
          
          {
            response?.map((eachResponse) => {
              return (
                <p>{eachResponse}</p>
              )
            })
          }
         
        </Row>
      </Container>}
    </Container>
  );
}

export default App;
