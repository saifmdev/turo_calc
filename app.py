import streamlit as st
import pandas as pd
import os
import streamlit.components.v1 as components
from fpdf import FPDF
from datetime import datetime
import openai

# Set page config
st.set_page_config(page_title="Turo Profitability Calculator", layout="wide")
st.title("🚗 Turo Vehicle Profitability Tool")

# Constants
DATA_FILE = "turo_vehicle_data.csv"
REFERENCE_CSV = os.path.join(os.path.dirname(__file__), "vehicle_makes_list.csv")
REPORTS_DIR = "reports"

# Script for copy button
COPY_BUTTON_SCRIPT = """
<script>
function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(function() {
    alert('Prompt copied to clipboard!');
  }, function(err) {
    alert('Failed to copy text: ', err);
  });
}
</script>
"""

def save_vehicle_entry(entry):
    if os.path.exists(DATA_FILE):
        existing = pd.read_csv(DATA_FILE)
        existing = existing[existing['Vehicle'] != entry['Vehicle']]
        updated = pd.concat([existing, pd.DataFrame([entry])], ignore_index=True)
    else:
        updated = pd.DataFrame([entry])
    updated.to_csv(DATA_FILE, index=False)

def load_saved_vehicles():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        columns = [
            "Vehicle", "Purchase Price", "Sale Price", "Months Owned", "Gross Daily Rate",
            "Rental Days/Month", "Turo Cut", "Yearly Maintenance", "Other Costs",
            "Initial KM", "Final KM", "KM Driven", "Net Income", "Operating Costs",
            "Depreciation", "Monthly Net Profit", "Annual ROI", "Profit per KM", "Verdict"
        ]
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(DATA_FILE, index=False)
        return empty_df

def generate_pdf_report(vehicle_data, prompt):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)

    try:
        pdf.image("logo.png", x=10, y=8, w=30)
    except:
        pass

    pdf.set_font("Arial", 'B', 16)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Turo Profitability Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Vehicle Details", ln=True)
    pdf.set_font("Arial", size=12)

    details = [
        ("Vehicle", vehicle_data['Vehicle']),
        ("Purchase Price", f"${vehicle_data['Purchase Price']:.2f}"),
        ("Sale Price", f"${vehicle_data['Sale Price']:.2f}"),
        ("Months Owned", str(vehicle_data['Months Owned'])),
        ("Gross Daily Rate", f"${vehicle_data['Gross Daily Rate']:.2f}"),
        ("Avg Rental Days/Month", f"{vehicle_data['Rental Days/Month']:.1f}"),
        ("Turo Cut", f"{vehicle_data['Turo Cut']*100:.1f}%"),
        ("Yearly Maintenance", f"${vehicle_data['Yearly Maintenance']:.2f}"),
        ("Other Costs", f"${vehicle_data['Other Costs']:.2f}"),
        ("KM Driven", f"{vehicle_data['KM Driven']:.0f} km"),
        ("Net Income", f"${vehicle_data['Net Income']:.2f}"),
        ("Operating Costs", f"${vehicle_data['Operating Costs']:.2f}"),
        ("Depreciation", f"${vehicle_data['Depreciation']:.2f}"),
        ("Monthly Net Profit", f"${vehicle_data['Monthly Net Profit']:.2f}"),
        ("Annual ROI", f"{vehicle_data['Annual ROI']*100:.2f}%"),
        ("Profit per KM", f"${vehicle_data['Profit per KM']:.2f}"),
        ("Verdict", vehicle_data['Verdict'])
    ]

    for label, value in details:
        pdf.cell(100, 10, txt=label + ":", ln=0)
        pdf.cell(90, 10, txt=value, ln=1)

    # Additional Breakdown
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Detailed Profitability Breakdown", ln=True)
    pdf.set_font("Arial", size=12)

    gross_income = vehicle_data['Gross Daily Rate'] * vehicle_data['Rental Days/Month'] * vehicle_data['Months Owned']
    turo_fee = gross_income * vehicle_data['Turo Cut']
    resale_roi = vehicle_data['Sale Price'] / vehicle_data['Purchase Price'] if vehicle_data['Purchase Price'] else 0
    break_even_months = (vehicle_data['Purchase Price'] - vehicle_data['Sale Price']) / max(vehicle_data['Monthly Net Profit'], 0.01)

    breakdown = [
        ("Total Gross Income", f"${gross_income:.2f}"),
        ("Total Turo Fees", f"${turo_fee:.2f}"),
        ("Resale ROI", f"{resale_roi*100:.2f}%"),
        ("Months to Break Even (w/ resale)", f"{break_even_months:.1f} months")
    ]

    for label, value in breakdown:
        pdf.cell(100, 10, txt=label + ":", ln=0)
        pdf.cell(90, 10, txt=value, ln=1)

    # AI Analysis
    pdf.add_page()
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="AI Profitability Analysis", ln=True)
    pdf.set_font("Arial", size=11)

    try:
        with st.spinner("Generating AI analysis for PDF..."):
            from openai import OpenAI
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You're an expert on vehicle reliability and Turo fleet strategy."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            chatgpt_reply = response.choices[0].message.content
            pdf.set_fill_color(240, 240, 255)
            pdf.multi_cell(0, 8, chatgpt_reply)
    except Exception as e:
        pdf.set_text_color(255, 0, 0)
        pdf.cell(200, 10, txt="Failed to generate AI analysis", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 8, f"Error: {str(e)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"Turo_Report_{vehicle_data['Vehicle'].replace(' ', '_')}_{timestamp}.pdf"
    report_path = os.path.join(REPORTS_DIR, report_filename)
    pdf.output(report_path)
    return report_path


# Main application
def main():
    # Sidebar - Vehicle selection and input
    st.sidebar.header("Enter or Select Vehicle Info")
    saved_data = load_saved_vehicles()
    vehicle_names = saved_data["Vehicle"].tolist() if not saved_data.empty else []
    selectable_list = ["New Vehicle"] + vehicle_names
    default_vehicle = st.session_state.get("selected_vehicle", "New Vehicle")
    selected_index = selectable_list.index(default_vehicle) if default_vehicle in selectable_list else 0
    selected_vehicle = st.sidebar.selectbox("Select a saved vehicle to load and edit:", selectable_list, index=selected_index)

    # Vehicle details input
    years = list(range(2024, 2012, -1))
    years = list(range(2024, 2012, -1))

    # Determine selected year from vehicle name if editing
    if selected_vehicle != "New Vehicle" and not saved_data.empty and selected_vehicle in saved_data["Vehicle"].values:
        parsed_parts = selected_vehicle.split()
        try:
            selected_year_val = int(parsed_parts[0])
            if selected_year_val not in years:
                selected_year_val = years[0]
        except:
            selected_year_val = years[0]
    else:
        selected_year_val = years[0]

    selected_year = st.sidebar.selectbox("Select Year", years, index=years.index(selected_year_val))


    try:
        makes_df = pd.read_csv(REFERENCE_CSV)
        makes = sorted(makes_df['Make'].unique())
    except Exception as e:
        st.sidebar.warning(f"Couldn't load makes list: {str(e)}")
        makes = []

    if selected_vehicle != "New Vehicle" and not saved_data.empty and selected_vehicle in saved_data["Vehicle"].values:
        # Load existing vehicle data
        selected_row = saved_data[saved_data["Vehicle"] == selected_vehicle].iloc[0].to_dict()
        parsed_parts = selected_row["Vehicle"].split()
        selected_make = parsed_parts[1] if len(parsed_parts) >= 2 else (makes[0] if makes else "")
        selected_make = st.sidebar.selectbox("Select Make", makes, index=makes.index(selected_make) if selected_make in makes else 0)
        
        entered_model = st.sidebar.text_input("Enter Model", value=selected_row["Vehicle"].split()[2] if len(selected_row["Vehicle"].split()) > 2 else "")
        entered_trim = st.sidebar.text_input("Enter Trim", value=' '.join(selected_row["Vehicle"].split()[3:]) if len(selected_row["Vehicle"].split()) > 3 else "")

        # Numeric inputs with existing values
        purchase_price = st.sidebar.number_input("Purchase Price ($)", value=float(selected_row["Purchase Price"]), min_value=0.0)
        sale_price = st.sidebar.number_input("Sale Price ($)", value=float(selected_row["Sale Price"]), min_value=0.0)
        months_owned = st.sidebar.number_input("Months Owned", value=int(selected_row["Months Owned"]), min_value=1)
        daily_rate = st.sidebar.number_input("Gross Daily Rental Rate ($)", value=float(selected_row["Gross Daily Rate"]), min_value=0.0)
        rental_days = st.sidebar.number_input("Avg Rental Days per Month", value=float(selected_row["Rental Days/Month"]), min_value=0.0)
        turo_cut = st.sidebar.slider("Turo Plan Cut (%)", 0.0, 1.0, value=float(selected_row["Turo Cut"]))
        maintenance_cost = st.sidebar.number_input("Yearly Maintenance Cost ($)", value=float(selected_row["Yearly Maintenance"]), min_value=0.0)
        other_costs = st.sidebar.number_input("Other Operating Costs (Total) ($)", value=float(selected_row["Other Costs"]), min_value=0.0)
        initial_km = st.sidebar.number_input("Initial KM", value=float(selected_row["Initial KM"]), min_value=0.0)
        final_km = st.sidebar.number_input("Final KM", value=float(selected_row["Final KM"]), min_value=0.0)
    else:
        # New vehicle inputs
        selected_make = st.sidebar.selectbox("Select Make", makes, index=0) if makes else st.sidebar.text_input("Enter Make")
        entered_model = st.sidebar.text_input("Enter Model")
        entered_trim = st.sidebar.text_input("Enter Trim")
        purchase_price = st.sidebar.number_input("Purchase Price ($)", min_value=0.0)
        sale_price = st.sidebar.number_input("Sale Price ($)", min_value=0.0)
        months_owned = st.sidebar.number_input("Months Owned", min_value=1)
        daily_rate = st.sidebar.number_input("Gross Daily Rental Rate ($)", min_value=0.0)
        rental_days = st.sidebar.number_input("Avg Rental Days per Month", min_value=0.0)
        turo_cut = st.sidebar.slider("Turo Plan Cut (%)", 0.0, 1.0, 0.25)
        maintenance_cost = st.sidebar.number_input("Yearly Maintenance Cost ($)", min_value=0.0)
        other_costs = st.sidebar.number_input("Other Operating Costs (Total) ($)", min_value=0.0)
        initial_km = st.sidebar.number_input("Initial KM", min_value=0.0)
        final_km = st.sidebar.number_input("Final KM", min_value=0.0)

    vehicle = f"{selected_year} {selected_make} {entered_model} {entered_trim}".strip()

    if st.sidebar.button("Calculate & Save Vehicle"):
        # Calculate all metrics
        total_gross_income = daily_rate * rental_days * months_owned
        turo_fee = total_gross_income * turo_cut
        net_income = total_gross_income - turo_fee
        total_maintenance = (months_owned / 12) * maintenance_cost
        total_operating = total_maintenance + other_costs
        depreciation = purchase_price - sale_price
        monthly_net_profit = (net_income - total_operating - depreciation) / months_owned
        km_driven = final_km - initial_km if final_km > initial_km else 0
        profit_per_km = (net_income - total_operating - depreciation) / km_driven if km_driven else 0
        annual_roi = ((net_income - total_operating - depreciation) / purchase_price) * (12 / months_owned)
        verdict = "Meets or Exceeds Target" if monthly_net_profit >= 525 * 0.7 else "Below Target"

        entry = {
            "Vehicle": vehicle,
            "Purchase Price": purchase_price,
            "Sale Price": sale_price,
            "Months Owned": months_owned,
            "Gross Daily Rate": daily_rate,
            "Rental Days/Month": rental_days,
            "Turo Cut": turo_cut,
            "Yearly Maintenance": maintenance_cost,
            "Other Costs": other_costs,
            "Initial KM": initial_km,
            "Final KM": final_km,
            "KM Driven": km_driven,
            "Net Income": net_income,
            "Operating Costs": total_operating,
            "Depreciation": depreciation,
            "Monthly Net Profit": monthly_net_profit,
            "Annual ROI": annual_roi,
            "Profit per KM": profit_per_km,
            "Verdict": verdict
        }

        if selected_vehicle != "New Vehicle":
            # Update existing entry
            index = saved_data[saved_data["Vehicle"] == selected_vehicle].index[0]
            for col in entry:
                saved_data.at[index, col] = entry[col]
            saved_data.to_csv(DATA_FILE, index=False)
        else:
            # Save new entry
            save_vehicle_entry(entry)
            saved_data = load_saved_vehicles()  # Reload data to include new entry

        st.success("Vehicle saved and analysis ready!")
        st.session_state["vehicle_data"] = entry  # Store for PDF generation

        # Create ChatGPT prompt (but don't show button)
        st.session_state["chatgpt_prompt"] = f"""I'll provide you with the year, make, model, and current kilometers of various vehicles one by one. 
For each, give me a thorough analysis of reliability, common issues, and suitability for Turo. 
At the end, tell me — if you were in my shoes, would you add it to your fleet? I will also provide you with current kilometers

{vehicle} with {initial_km} purchase price {purchase_price}. 

keep the response as brief as possible
"""

        # Display results
        st.subheader("📊 Vehicle Profitability Summary")
        st.markdown(f"**Vehicle:** {vehicle}")
        st.markdown(f"**Monthly Net Profit:** ${monthly_net_profit:.2f}")
        st.markdown(f"**Annual ROI:** {annual_roi*100:.2f}%")
        st.markdown(f"**Profit per KM:** ${profit_per_km:.2f}")
        st.markdown(f"**Net Income:** ${net_income:.2f}")
        st.markdown(f"**Operating Costs:** ${total_operating:.2f}")
        st.markdown(f"**Depreciation:** ${depreciation:.2f}")
        st.markdown(f"**Verdict:** {verdict}")

        # PDF Report Generation Button (MUST be outside the Save button block)
    if st.button("Generate PDF Report"):
        with st.spinner("Generating PDF report..."):
            try:
                if "vehicle_data" not in st.session_state or "chatgpt_prompt" not in st.session_state:
                    st.error("⚠️ No vehicle data available to generate a report.")
                else:
                    entry = st.session_state["vehicle_data"]
                    prompt = st.session_state["chatgpt_prompt"]

                    report_path = generate_pdf_report(entry, prompt)

                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=f,
                            file_name=os.path.basename(report_path),
                            mime="application/pdf"
                        )

                    st.success("✅ PDF report generated successfully!")
            except Exception as e:
                st.error(f"❌ Failed to generate PDF: {e}")

    # Saved Vehicles Section
    st.write("---")
    st.subheader("📁 Saved Vehicle Entries")
    if not saved_data.empty:
        for i, row in saved_data.iterrows():
            with st.expander(f"{row['Vehicle']}"):
                st.write(f"**Monthly Net Profit:** ${row['Monthly Net Profit']:.2f}")
                st.write(f"**Annual ROI:** {row['Annual ROI']*100:.2f}%")
                st.write(f"**Profit per KM:** ${row['Profit per KM']:.2f}")
                verdict_icon = "✅" if row['Verdict'] == "Meets or Exceeds Target" else "⚠️"
                st.write(f"**Verdict:** {verdict_icon} {row['Verdict']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Update", key=f"update_{i}"):
                        st.session_state["selected_vehicle"] = row['Vehicle']
                        st.rerun()
                with col2:
                    if st.button("Delete", key=f"delete_{i}"):
                        saved_data = saved_data.drop(i)
                        saved_data.to_csv(DATA_FILE, index=False)
                        st.success(f"Deleted {row['Vehicle']}")
                        st.rerun()

        # st.dataframe(saved_data[["Vehicle", "Monthly Net Profit", "Annual ROI", "Profit per KM", "Verdict"]].reset_index(drop=True))
    else:
        st.info("No vehicles saved yet.")

if __name__ == "__main__":
    main()
    
    