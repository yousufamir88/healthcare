# Retrieval comparison

Document-level precision and recall for the two chunking strategies.
Multiple chunks from the same parent document count once.

Top-k chunks: 3

| Query | Strategy | Expected | Retrieved | TP | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| How do I book a clinic appointment? | fixed_size_overlap | appointment-booking | appointment-booking, insurance-claims, emergency-protocol | 1 | 0.3333 | 1.0000 |
| How do I book a clinic appointment? | sentence | appointment-booking | appointment-booking, emergency-protocol, telemedicine | 1 | 0.3333 | 1.0000 |
| What is the cancellation and rescheduling policy? | fixed_size_overlap | cancellation-rescheduling | cancellation-rescheduling | 1 | 1.0000 | 1.0000 |
| What is the cancellation and rescheduling policy? | sentence | cancellation-rescheduling | cancellation-rescheduling | 1 | 1.0000 | 1.0000 |
| What are the consultation fees by specialty? | fixed_size_overlap | consultation-fees | follow-up-discount, appointment-booking, home-visit | 0 | 0.0000 | 0.0000 |
| What are the consultation fees by specialty? | sentence | consultation-fees | follow-up-discount, consultation-fees, appointment-booking | 1 | 0.3333 | 1.0000 |
| How does the insurance claim process work? | fixed_size_overlap | insurance-claims | insurance-claims | 1 | 1.0000 | 1.0000 |
| How does the insurance claim process work? | sentence | insurance-claims | insurance-claims, prescription-refills | 1 | 0.5000 | 1.0000 |
| What is the follow-up discount? | fixed_size_overlap | follow-up-discount | follow-up-discount | 1 | 1.0000 | 1.0000 |
| What is the follow-up discount? | sentence | follow-up-discount | follow-up-discount | 1 | 1.0000 | 1.0000 |
| Who is eligible for a telemedicine video visit? | fixed_size_overlap | telemedicine | telemedicine | 1 | 1.0000 | 1.0000 |
| Who is eligible for a telemedicine video visit? | sentence | telemedicine | telemedicine | 1 | 1.0000 | 1.0000 |
| What is the cardiology consultation fee and can a first cardiology visit be done by video? | fixed_size_overlap | consultation-fees, telemedicine | follow-up-discount, home-visit, consultation-fees | 1 | 0.3333 | 0.5000 |
| What is the cardiology consultation fee and can a first cardiology visit be done by video? | sentence | consultation-fees, telemedicine | follow-up-discount, telemedicine | 1 | 0.5000 | 0.5000 |

## Arithmetic

- **fixed_size_overlap** — How do I book a clinic appointment?
  - Precision: 1 / 3 = 0.3333333333333333
  - Recall: 1 / 1 = 1.0
- **sentence** — How do I book a clinic appointment?
  - Precision: 1 / 3 = 0.3333333333333333
  - Recall: 1 / 1 = 1.0
- **fixed_size_overlap** — What is the cancellation and rescheduling policy?
  - Precision: 1 / 1 = 1.0
  - Recall: 1 / 1 = 1.0
- **sentence** — What is the cancellation and rescheduling policy?
  - Precision: 1 / 1 = 1.0
  - Recall: 1 / 1 = 1.0
- **fixed_size_overlap** — What are the consultation fees by specialty?
  - Precision: 0 / 3 = 0.0
  - Recall: 0 / 1 = 0.0
- **sentence** — What are the consultation fees by specialty?
  - Precision: 1 / 3 = 0.3333333333333333
  - Recall: 1 / 1 = 1.0
- **fixed_size_overlap** — How does the insurance claim process work?
  - Precision: 1 / 1 = 1.0
  - Recall: 1 / 1 = 1.0
- **sentence** — How does the insurance claim process work?
  - Precision: 1 / 2 = 0.5
  - Recall: 1 / 1 = 1.0
- **fixed_size_overlap** — What is the follow-up discount?
  - Precision: 1 / 1 = 1.0
  - Recall: 1 / 1 = 1.0
- **sentence** — What is the follow-up discount?
  - Precision: 1 / 1 = 1.0
  - Recall: 1 / 1 = 1.0
- **fixed_size_overlap** — Who is eligible for a telemedicine video visit?
  - Precision: 1 / 1 = 1.0
  - Recall: 1 / 1 = 1.0
- **sentence** — Who is eligible for a telemedicine video visit?
  - Precision: 1 / 1 = 1.0
  - Recall: 1 / 1 = 1.0
- **fixed_size_overlap** — What is the cardiology consultation fee and can a first cardiology visit be done by video?
  - Precision: 1 / 3 = 0.3333333333333333
  - Recall: 1 / 2 = 0.5
- **sentence** — What is the cardiology consultation fee and can a first cardiology visit be done by video?
  - Precision: 1 / 2 = 0.5
  - Recall: 1 / 2 = 0.5

## Macro scores

- fixed_size_overlap: precision 0.6667, recall 0.7857, F1 0.7213
- sentence: precision 0.6667, recall 0.9286, F1 0.7761

## Recommendation

Use **sentence**.

sentence has macro F1 0.7761 (precision 0.6667, recall 0.9286) compared with fixed_size_overlap macro F1 0.7213 (precision 0.6667, recall 0.7857). Each macro score is the unweighted mean across the evaluation queries. F1 is the harmonic mean of that macro precision and macro recall.
