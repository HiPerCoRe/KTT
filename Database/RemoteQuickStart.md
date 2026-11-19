# Run the remote example on supercuda

Five steps to store tuning results on the ktt-db server at `https://supercuda.fi.muni.cz:8443`.
Commands are for Linux and start from the KTT root folder.

## 1. Connect to the FI MUNI VPN

The server can only be reached from the FI MUNI network. Without the VPN, every request fails with
"Couldn't connect to server" or "Timeout was reached".

## 2. Build the example

```bash
sudo apt install libcurl4-openssl-dev           # once: the remote database needs libcurl
./premake5 gmake --remote-database --no-opencl  # supercuda is the default server
cd Build
make -j$(nproc) CoulombSum3dRemoteCuda
```

## 3. Save your API key

1. Open `https://supercuda.fi.muni.cz:8443` in a browser and sign in. The browser warns about the certificate
   (see step 4); accept it.
2. Open your profile menu → **API keys** and create a key with the **write** permission (the example saves results).
3. Save the key to the key file. KTT reads it from there, so you never pass it on the command line:

```bash
mkdir -p ~/.local/share/ktt
nano ~/.local/share/ktt/api-key        # paste the key, save, exit
chmod 600 ~/.local/share/ktt/api-key   # only you can read it
```

The file holds just the key, nothing else. On Windows the file is `%LOCALAPPDATA%\ktt\api-key`.

## 4. Download the server's certificate

**Why:** the connection is encrypted (HTTPS), and KTT checks the server's certificate to be sure it talks to the real
server. The certificate of supercuda is *self-signed*: no public certificate authority vouches for it, so KTT does not
trust it by default and stops with:

```
SSL peer certificate or SSH remote key was not OK (SSL certificate problem: self-signed certificate)
```

The fix is to download the certificate once and tell KTT to trust it. Run this **on the VPN**:

```bash
openssl s_client -connect supercuda.fi.muni.cz:8443 -servername supercuda.fi.muni.cz </dev/null 2>/dev/null | openssl x509 > ~/supercuda.pem
```

Check that it worked:

```bash
openssl x509 -in ~/supercuda.pem -noout -subject   # prints: subject=CN = supercuda
```

> **Note:** `~/supercuda.pem` is only an example path. You can save the file anywhere and name it however you like.
> If you change the path, use the same path in `--ca-bundle` in step 5.

When the server gets a new certificate, the old file stops working: download it again.

## 5. Run the example

Run it from `Build/x86_64_Release`, because the example loads its kernels relative to that folder:

```bash
cd Build/x86_64_Release
./CoulombSum3dRemoteCuda --ca-bundle ~/supercuda.pem
```

It prints `Using ktt-db server https://supercuda.fi.muni.cz:8443`, tunes the kernel and saves the results to the
server. Look at them in the web app.

## When something fails

| Message contains | What to do |
|---|---|
| `Couldn't connect to server`, `Timeout was reached`, `Could not resolve host` | Connect to the FI MUNI VPN (step 1), or the server is down. |
| `self-signed certificate` | You forgot `--ca-bundle`, or its path is wrong (step 4). |
| `no alternative certificate subject name matches` | The server's certificate does not include `supercuda.fi.muni.cz`. Tell the server admin. |
| `the API key file ... does not exist` / `is empty` | Save the key to `~/.local/share/ktt/api-key` (step 3). |
| `HTTP 401` | The key is wrong or was deleted: create a new one (step 3). |
| `HTTP 403` | The key has only the read permission, or no realm: create one with **write** and a realm. |

## Bonus: the integration test

The same server and certificate also work for the integration test of the database client:

```bash
./premake5 gmake --remote-database --no-opencl --tests
cd Build && make -j$(nproc) Tests && cd x86_64_Release
KTT_DB_CA_BUNDLE=~/supercuda.pem KTT_DB_API_KEY=$(cat ~/.local/share/ktt/api-key) ./Tests "[integration]"
```
