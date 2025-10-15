from fpdf import FPDF
import os

os.makedirs("demo_pdfs", exist_ok=True)

# -------------------------
# 1️⃣ Company Policy PDF
# -------------------------
company_policy_text = """
Travel Policy:

1. Employees may travel by bus or train only. Flights are not reimbursed.
2. Travel allowance up to INR 500 per trip.
3. Hotel stays up to INR 1000 per night allowed only for 3-star hotels.
"""

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
for line in company_policy_text.strip().split("\n"):
    pdf.multi_cell(0, 10, line)
company_policy_path = "demo_pdfs/company_policy_demo.pdf"
pdf.output(company_policy_path)
print(f"Generated company policy PDF: {company_policy_path}")

# -------------------------
# 2️⃣ Client Bill PDF - Compliant
# -------------------------
compliant_bill_text = """
Bill - Travel

Employee traveled by bus from Hyderabad to Bangalore.
Total fare: INR 500.

"""

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
for line in compliant_bill_text.strip().split("\n"):
    pdf.multi_cell(0, 10, line)
compliant_bill_path = "demo_pdfs/client_bill_compliant.pdf"
pdf.output(compliant_bill_path)
print(f"Generated compliant client bill PDF: {compliant_bill_path}")

# -------------------------
# 3️⃣ Client Bill PDF - Violating
# -------------------------
violating_bill_text = """
Bill - Travel

Employee traveled by flight from Hyderabad to Bangalore.
Total fare: INR 800.

"""

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
for line in violating_bill_text.strip().split("\n"):
    pdf.multi_cell(0, 10, line)
violating_bill_path = "demo_pdfs/client_bill_violating.pdf"
pdf.output(violating_bill_path)
print(f"Generated violating client bill PDF: {violating_bill_path}")



# -------------------------
# 3️⃣ Client Bill PDF - Violating
# -------------------------
violating_bill_text = """
Bill - stay

Hotel stay at a 5-star hotel for 2 nights.
Total: INR 1800.
"""

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
for line in violating_bill_text.strip().split("\n"):
    pdf.multi_cell(0, 10, line)
violating_bill_path = "demo_pdfs/client_bill_violating1.pdf"
pdf.output(violating_bill_path)
print(f"Generated violating client bill PDF: {violating_bill_path}")
