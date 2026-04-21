import pandas as pd
import os
from datetime import datetime
from vector_db import VectorDB

class SalesRAG:
    def __init__(self):
        self.db = VectorDB()
        self.data_path = "./data"
        self._enquiry_df = None
        self._appointment_df = None
        self._feedback_df = None

    def _load_dataframes(self):
        """Load and cache all three dataframes for structured queries."""
        if self._enquiry_df is None:
            self._enquiry_df = pd.read_csv(
                os.path.join(self.data_path, "sales_enquiry_dataset.csv")
            ).fillna("Unknown")
            # Normalize date column
            self._enquiry_df["Enquiry Date"] = pd.to_datetime(
                self._enquiry_df["Enquiry Date"], errors="coerce"
            )
            self._enquiry_df["Appointment Date"] = pd.to_datetime(
                self._enquiry_df["Appointment Date"], errors="coerce"
            )

        if self._appointment_df is None:
            self._appointment_df = pd.read_csv(
                os.path.join(self.data_path, "sales_appointment_dataset.csv")
            ).fillna("Unknown")
            self._appointment_df["Appointment Date"] = pd.to_datetime(
                self._appointment_df["Appointment Date"], errors="coerce"
            )

        if self._feedback_df is None:
            self._feedback_df = pd.read_csv(
                os.path.join(self.data_path, "sales_feedback_dataset.csv")
            ).fillna("Unknown")
            self._feedback_df["Date"] = pd.to_datetime(
                self._feedback_df["Date"], errors="coerce"
            )

    def load_and_process_csvs(self):
        """Load all 3 CSVs and convert rows into meaningful sentences for the vector DB."""
        self._load_dataframes()
        documents = []
        ids = []
        metadatas = []

        # ---------------------------------------------------------
        # Process Enquiry CSV — correct column: 'Vehicle Name / Model', 'Test Ride Taken'
        # ---------------------------------------------------------
        for idx, row in self._enquiry_df.iterrows():
            sentence = (
                f"Customer {row['Customer Name']} (Phone: {row['Phone Number']}) "
                f"made an enquiry for {row['Vehicle Name / Model']} via {row['Enquiry Source']} "
                f"on {row['Enquiry Date'].strftime('%d/%m/%Y') if pd.notna(row['Enquiry Date']) else 'Unknown'}. "
                f"Status: {row['Status']}. Budget/Payment: {row['Payment Type']}. "
                f"Test ride taken: {row['Test Ride Taken']}. "
                f"Customer type: {row['Customer Type']}. City: {row['City / State']}."
            )
            documents.append(sentence)
            ids.append(f"enq_{idx}")
            metadatas.append({"source": "enquiry", "status": str(row["Status"])})

        # ---------------------------------------------------------
        # Process Appointment CSV — correct column: 'Vehicle'
        # ---------------------------------------------------------
        for idx, row in self._appointment_df.iterrows():
            sentence = (
                f"Appointment for customer {row['Customer Name']} "
                f"for vehicle {row['Vehicle']} on "
                f"{row['Appointment Date'].strftime('%d/%m/%Y') if pd.notna(row['Appointment Date']) else 'Unknown'} "
                f"at {row['Time']}. Status: {row['Status']}."
            )
            documents.append(sentence)
            ids.append(f"app_{idx}")
            metadatas.append({"source": "appointment", "status": str(row["Status"])})

        # ---------------------------------------------------------
        # Process Feedback CSV — correct column: 'Date' (not 'Feedback Date')
        # ---------------------------------------------------------
        for idx, row in self._feedback_df.iterrows():
            sentence = (
                f"Customer {row['Customer Name']} gave feedback: '{row['Feedback']}'. "
                f"Rating: {row['Rating']}/5 on "
                f"{row['Date'].strftime('%d/%m/%Y') if pd.notna(row['Date']) else 'Unknown'}."
            )
            documents.append(sentence)
            ids.append(f"fb_{idx}")
            metadatas.append({"source": "feedback"})

        print(f"✅ Processed {len(documents)} documents from 3 CSV files.")
        self.db.add_documents(documents=documents, ids=ids, metadatas=metadatas)

    def search_similar(self, query: str, n_results: int = 6):
        """Search for similar past sales records via vector similarity."""
        return self.db.search(query, n_results)

    # ------------------------------------------------------------------
    # STRUCTURED QUERY METHODS  (bypass RAG, use pandas directly)
    # These give exact counts / summaries instead of fuzzy text chunks.
    # ------------------------------------------------------------------

    def get_structured_context(self, query: str) -> str | None:
        """
        Detects if the query needs a precise data answer (counts, date filters,
        model-specific stats) and returns a pre-built context string.
        Returns None if the query is better handled by vector search.
        """
        self._load_dataframes()
        q = query.lower()
        today = datetime.today().date()

        # ---- DATE-BASED APPOINTMENT QUERIES ----
        target_date = None
        if "today" in q:
            target_date = today
        elif "tomorrow" in q:
            import datetime as dt
            target_date = today + dt.timedelta(days=1)
        elif "yesterday" in q:
            import datetime as dt
            target_date = today - dt.timedelta(days=1)
        else:
            # Try to parse a date from the query  e.g. "21/04/2026" or "04/21/2026"
            import re
            date_patterns = [
                r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b",
                r"\b(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})\b",
            ]
            for pat in date_patterns:
                m = re.search(pat, q)
                if m:
                    try:
                        raw = m.group(0).replace("-", "/")
                        parts = raw.split("/")
                        if len(parts[2]) == 4:          # dd/mm/yyyy or mm/dd/yyyy
                            # Try mm/dd/yyyy first (US style like in CSV), then dd/mm
                            for fmt in ["%m/%d/%Y", "%d/%m/%Y"]:
                                try:
                                    target_date = datetime.strptime(raw, fmt).date()
                                    break
                                except ValueError:
                                    pass
                        else:                           # yyyy/mm/dd
                            target_date = datetime.strptime(raw, "%Y/%m/%d").date()
                    except Exception:
                        pass
                    break

        # Appointment / report for a specific date
        if target_date and any(w in q for w in ["appointment", "schedule", "report", "meeting", "today", "tomorrow", "yesterday"]):
            df = self._appointment_df.copy()
            df["_date"] = df["Appointment Date"].dt.date
            filtered = df[df["_date"] == target_date]
            date_str = target_date.strftime("%d/%m/%Y")
            if filtered.empty:
                return f"No appointments found for {date_str}."
            lines = [f"Appointments for {date_str} (Today: {today.strftime('%d/%m/%Y')}):"]
            for _, row in filtered.iterrows():
                lines.append(
                    f"  - {row['Customer Name']} | {row['Vehicle']} | {row['Time']} | Status: {row['Status']}"
                )
            lines.append(f"\nTotal: {len(filtered)} appointment(s).")
            return "\n".join(lines)

        # ---- COUNT QUERIES ----
        if any(w in q for w in ["how many", "count", "total", "number of"]):

            # Enquiries on a date
            if target_date and "enquir" in q:
                df = self._enquiry_df.copy()
                df["_date"] = df["Enquiry Date"].dt.date
                c = len(df[df["_date"] == target_date])
                return f"Total enquiries on {target_date.strftime('%d/%m/%Y')}: {c}"

            # Cars booked/sold
            if any(w in q for w in ["sold", "booked", "purchased", "bought", "sale"]):
                if target_date:
                    df = self._enquiry_df.copy()
                    df["_date"] = df["Enquiry Date"].dt.date
                    c = len(df[(df["_date"] == target_date) & (df["Status"].str.lower() == "booked")])
                    return f"Bookings (status=Booked) on {target_date.strftime('%d/%m/%Y')}: {c}"
                # All-time
                c = len(self._enquiry_df[self._enquiry_df["Status"].str.lower() == "booked"])
                return f"Total bookings (status=Booked) in the database: {c}"

            # Appointments
            if "appointment" in q:
                if target_date:
                    df = self._appointment_df.copy()
                    df["_date"] = df["Appointment Date"].dt.date
                    c = len(df[df["_date"] == target_date])
                    return f"Total appointments on {target_date.strftime('%d/%m/%Y')}: {c}"
                total = len(self._appointment_df)
                scheduled = len(self._appointment_df[self._appointment_df["Status"].str.lower() == "scheduled"])
                cancelled = len(self._appointment_df[self._appointment_df["Status"].str.lower() == "cancelled"])
                completed = len(self._appointment_df[self._appointment_df["Status"].str.lower() == "completed"])
                return (
                    f"Appointment summary:\n"
                    f"  Total: {total}\n"
                    f"  Scheduled: {scheduled}\n"
                    f"  Completed: {completed}\n"
                    f"  Cancelled: {cancelled}"
                )

        # ---- VEHICLE-SPECIFIC QUERIES ----
        # Try to detect a vehicle model name in the query
        all_vehicles = pd.concat([
            self._enquiry_df["Vehicle Name / Model"],
            self._appointment_df["Vehicle"],
        ]).dropna().unique()

        matched_vehicle = None
        for v in all_vehicles:
            if str(v).lower() in q:
                matched_vehicle = str(v)
                break

        if matched_vehicle:
            enq = self._enquiry_df[
                self._enquiry_df["Vehicle Name / Model"].str.lower() == matched_vehicle.lower()
            ]
            appt = self._appointment_df[
                self._appointment_df["Vehicle"].str.lower() == matched_vehicle.lower()
            ]
            fb_keywords = matched_vehicle.lower().split()

            lines = [f"Data summary for vehicle: {matched_vehicle}"]
            lines.append(f"\nEnquiries: {len(enq)}")
            if not enq.empty:
                for status, grp in enq.groupby("Status"):
                    lines.append(f"  - {status}: {len(grp)}")
            lines.append(f"\nAppointments: {len(appt)}")
            if not appt.empty:
                for status, grp in appt.groupby("Status"):
                    lines.append(f"  - {status}: {len(grp)}")

            # Feedback — search by keyword since feedback CSV has no vehicle column
            fb = self._feedback_df[
                self._feedback_df["Feedback"].str.lower().str.contains(fb_keywords[0], na=False)
            ]
            if not fb.empty:
                avg_rating = fb["Rating"].mean()
                lines.append(f"\nFeedback mentions: {len(fb)}")
                lines.append(f"Average rating: {avg_rating:.1f}/5")
                for _, row in fb.iterrows():
                    lines.append(f"  - {row['Customer Name']}: '{row['Feedback']}' ({row['Rating']}/5)")

            return "\n".join(lines)

        # No structured match — fall back to vector search
        return None


# For testing / initial load
if __name__ == "__main__":
    rag = SalesRAG()
    rag.load_and_process_csvs()
    print("RAG setup completed! Data is safely stored in ChromaDB.")