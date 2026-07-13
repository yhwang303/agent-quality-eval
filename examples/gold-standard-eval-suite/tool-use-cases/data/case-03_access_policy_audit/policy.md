# Access Policy

Apply these rules in order:

1. Terminated users must have zero access to all resources.
2. A user's department must appear in the resource's `allowed_departments`.
3. Contractors are allowed only when `allow_contractors` is true and the user's department is also allowed.
4. Restricted resources:
   - `admin` permission requires a ticket starting with `SEC-`.
   - `export` permission requires a ticket starting with `DPO-`.
5. Confidential resources:
   - `write` permission requires a ticket starting with `CHG-`.
6. Internal resources:
   - `export` is allowed for allowed departments.

Severity guidance:

- Critical: terminated-user access or invalid admin access to restricted resources.
- High: contractor access where contractors are not allowed, or department mismatch on confidential/restricted resources.
- Medium: department mismatch on internal resources.
- Low: documentation or ticket-prefix issue that does not expose restricted or confidential resources.
