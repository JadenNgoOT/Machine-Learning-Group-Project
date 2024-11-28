import React, { useState } from 'react';
import { Form, Container, Row, Col, Card, Button } from 'react-bootstrap';


const homePage = () => {
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    weight: '',
    height: '',
    sleepHours: '',
    stressLevel: '',
    exerciseFrequency: '',
    caffeineIntake: '',
    alcoholConsumption: '',
    screenTimeBeforeSleep: '',
    bedtime: '',
    wakeupTime: '',
    medicalConditions: '',
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevState => ({
      ...prevState,
      [name]: value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Here you would typically send the data to your ML model
    console.log('Form submitted:', formData);
  };

  return (
    <Container className="mt-5">
      <Card>
        <Card.Header as="h2" className="text-center bg-primary text-white">
          Sleep Quality Predictor
        </Card.Header>
        <Card.Body>
          <Form onSubmit={handleSubmit}>
            <Row>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Age</Form.Label>
                  <Form.Control 
                    type="number" 
                    name="age"
                    value={formData.age}
                    onChange={handleChange}
                    placeholder="Enter your age"
                    required
                  />
                </Form.Group>
              </Col>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Gender</Form.Label>
                  <Form.Select 
                    name="gender"
                    value={formData.gender}
                    onChange={handleChange}
                    required
                  >
                    <option value="">Select Gender</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="other">Other</option>
                  </Form.Select>
                </Form.Group>
              </Col>
            </Row>

            <Row>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Weight (kg)</Form.Label>
                  <Form.Control 
                    type="number" 
                    name="weight"
                    value={formData.weight}
                    onChange={handleChange}
                    placeholder="Enter your weight"
                    required
                  />
                </Form.Group>
              </Col>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Height (cm)</Form.Label>
                  <Form.Control 
                    type="number" 
                    name="height"
                    value={formData.height}
                    onChange={handleChange}
                    placeholder="Enter your height"
                    required
                  />
                </Form.Group>
              </Col>
            </Row>

            <Row>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Average Sleep Hours</Form.Label>
                  <Form.Control 
                    type="number" 
                    name="sleepHours"
                    value={formData.sleepHours}
                    onChange={handleChange}
                    placeholder="Hours of sleep per night"
                    min="0"
                    max="24"
                    required
                  />
                </Form.Group>
              </Col>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Stress Level</Form.Label>
                  <Form.Select 
                    name="stressLevel"
                    value={formData.stressLevel}
                    onChange={handleChange}
                    required
                  >
                    <option value="">Select Stress Level</option>
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                  </Form.Select>
                </Form.Group>
              </Col>
            </Row>

            <Row>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Exercise Frequency</Form.Label>
                  <Form.Select 
                    name="exerciseFrequency"
                    value={formData.exerciseFrequency}
                    onChange={handleChange}
                    required
                  >
                    <option value="">Select Frequency</option>
                    <option value="never">Never</option>
                    <option value="1-2">1-2 times/week</option>
                    <option value="3-4">3-4 times/week</option>
                    <option value="5+">5+ times/week</option>
                  </Form.Select>
                </Form.Group>
              </Col>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Caffeine Intake</Form.Label>
                  <Form.Select 
                    name="caffeineIntake"
                    value={formData.caffeineIntake}
                    onChange={handleChange}
                    required
                  >
                    <option value="">Select Intake</option>
                    <option value="none">None</option>
                    <option value="1-2">1-2 cups/day</option>
                    <option value="3-4">3-4 cups/day</option>
                    <option value="5+">5+ cups/day</option>
                  </Form.Select>
                </Form.Group>
              </Col>
            </Row>

            <Row>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Alcohol Consumption</Form.Label>
                  <Form.Select 
                    name="alcoholConsumption"
                    value={formData.alcoholConsumption}
                    onChange={handleChange}
                    required
                  >
                    <option value="">Select Consumption</option>
                    <option value="none">None</option>
                    <option value="occasional">Occasional</option>
                    <option value="regular">Regular</option>
                  </Form.Select>
                </Form.Group>
              </Col>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Screen Time Before Sleep (mins)</Form.Label>
                  <Form.Control 
                    type="number" 
                    name="screenTimeBeforeSleep"
                    value={formData.screenTimeBeforeSleep}
                    onChange={handleChange}
                    placeholder="Minutes of screen time"
                    min="0"
                    required
                  />
                </Form.Group>
              </Col>
            </Row>

            <Row>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Typical Bedtime</Form.Label>
                  <Form.Control 
                    type="time" 
                    name="bedtime"
                    value={formData.bedtime}
                    onChange={handleChange}
                    required
                  />
                </Form.Group>
              </Col>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Wake-up Time</Form.Label>
                  <Form.Control 
                    type="time" 
                    name="wakeupTime"
                    value={formData.wakeupTime}
                    onChange={handleChange}
                    required
                  />
                </Form.Group>
              </Col>
            </Row>

            <Form.Group className="mb-3">
              <Form.Label>Medical Conditions</Form.Label>
              <Form.Control 
                as="textarea" 
                name="medicalConditions"
                value={formData.medicalConditions}
                onChange={handleChange}
                placeholder="List any relevant medical conditions"
                rows={3}
              />
            </Form.Group>

            <div className="text-center">
              <Button variant="primary" type="submit" size="lg">
                Predict Sleep Quality
              </Button>
            </div>
          </Form>
        </Card.Body>
      </Card>
    </Container>
  );
};

export default homePage;